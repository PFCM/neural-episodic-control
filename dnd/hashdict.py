"""The differentiable neural dictionary. Or at least, something inspired
by it.

Problems with this approach:
    - accuracy of lookup
        - going to be highly dependent on the similarities, might be quite
          blurry, meaning that if you lookup up an exact key, you'll still get
          an average of its neighbours
        - probably could softmax with a temperature to make it nice and spiky?
    - probably more...
"""
import numpy as np
import tensorflow as tf

from dnd.lsh import simhash, get_simhash_config
import dnd.similarities


class HashDND(object):
    """differentiable neural dictionary, using LSH for approximate
    nearest neighbour lookup. Assumes keys are vectors. Also assumes we only
    use float32 and doesn't handle batched operations :("""

    sentinel_value = np.inf

    @classmethod
    def _setup_variables(cls, hash_bits, max_neighbours, key_size,
                         value_shapes):
        """setup variables with appropriate initializers given the shapes"""
        init = tf.constant_initializer(cls.sentinel_value)

        keys = tf.get_variable(name='keys',
                               shape=[2**hash_bits * max_neighbours, key_size],
                               initializer=init)

        values = []
        for i, shape in enumerate(value_shapes):
            var_shape = [2**hash_bits * max_neighbours] + shape
            var = tf.get_variable(name='value_{}'.format(i),
                                  shape=var_shape,
                                  initializer=init)
            values.append(var)
        return keys, values

    def __init__(self, hash_bits, max_neighbours, key_size, value_shapes,
                 similarity_measure=None):
        """Set up the dnd.

        Args:
            hash_bits (int): how many bits for the hash. There will be
                `2**num_bits` individual buckets.
            max_neighbours (int): how many entries to store in each bucket.
                This controls the number of neighbours we check against.
                Operations will be linear in this value and it will likely
                effect learning performance significantly as well.
            key_size (int): size of the key vectors. We use the unhashed key
                vectors to compute similarities between keys we find from the
                nearest neighbour lookup.
            value_shapes (list): list of shapes for the values stored in the
                dictionary.
            similarity_measure (Optional[callable]): function which adds ops
                to compare a query key with all of the other keys in the
                bucket. If unspecified, the cosine similarity is used. Should
                be a callable which takes two input tensors: the query key
                (shaped `[key_size]`) and a  `[max_neighbours, key_size]`
                tensor  of keys to compare against. Should return a
                `[max_neighbours]` tensor of similarities, between 0 and 1
                where 1 means the two keys were identical.
        """
        self._hash_size = hash_bits
        self._key_size = key_size
        self._bucket_size = max_neighbours
        self._keys, self._values = HashDND._setup_variables(hash_bits,
                                                            max_neighbours,
                                                            key_size,
                                                            value_shapes)
        self._hash_config = get_simhash_config(self._key_size,
                                               self._hash_size)

        if not similarity_measure:
            similarity_measure = dnd.similarities.cosine_similarity
        self._similarity_measure = similarity_measure
        self._summarise_pressure()

    def _summarise_pressure(self):
        """add summaries for the load. It would be nice to have this
        per-bucket, but there is potentially a lot of buckets."""
        with tf.name_scope('dnd_stats'):
            used_keys = tf.not_equal(self._keys[:, 0], self.sentinel_value)
            filled_keys = tf.reduce_sum(tf.cast(used_keys, tf.float32))
            tf.summary.scalar('total_fill', filled_keys)

    def _get_bucket(self, key):
        """look up the contents of a bucket by hash. Also return the bucket
        index so we can create updates to the storage variables."""
        idx = simhash(key, self._hash_config)
        bucket_start = idx * self._bucket_size
        bucket_end = (idx + 1) * self._bucket_size
        keys = self._keys[bucket_start:bucket_end, ...]
        values = [val[bucket_start:bucket_end, ...] for val in self._values]
        return keys, values, idx

    def store(self, key, value):
        """Gets an op which will store the key-value pair. This involves the
        following process:
            - compute hash of `key`
            - lookup all keys and values with matching hash
            - if the bucket isn't full
                - assign the values to the values of the next empty position
            - else (if the bucket is full)
                - update, according to some update rule which may be
                  application specific.
                - TODO: figure this out (LRU?)

        Args:
            key (tensor): `[key_size]` key to store
            value (list of tensors): `[???]` tensors to
                be stored.

        Returns:
            op: an op which carries out the above steps.
        """
        with tf.name_scope('dnd/store'):
            bucket_keys, bucket_values, idx = self._get_bucket(key)
            # is there space?
            can_store = tf.reduce_any(tf.equal(bucket_keys[:, 0],
                                               self.sentinel_value))

            def _empty_store():
                return self._get_store_op_empty(key, value, idx, bucket_keys)

            def _full_store():
                return self._get_store_op_full(key, value, idx, bucket_keys)

            store_op = tf.cond(can_store, _empty_store, _full_store)
        return store_op

    def _flatten_index(self, index, bucket_index):
        """turn a bucket-level index into a global index"""
        return index + (bucket_index * self._bucket_size)

    def _update_at_index(self, index, new_key, new_vals):
        """make update ops to insert at the appropriate (flattened) index"""
        # update the keys
        key_update = tf.scatter_update(self._keys, [index],
                                       tf.expand_dims(new_key, 0))
        # and update the values
        value_updates = []
        for value_var, new_val in zip(self._values, new_vals):
            val_update = tf.scatter_update(value_var, [index],
                                           tf.expand_dims(new_val, 0))
            value_updates.append(val_update)

        # make sure they all happen at once
        return tf.group(key_update, *value_updates)

    def _get_store_op_empty(self, store_key, store_vals, bucket_index,
                            bucket_keys):
        """get an op to store given key and values in the first empty space.

        Returns an op with no output that will run all of the required updates.
        """
        # first find the first empty spot (assuming there is one)
        with tf.name_scope('empty_store'):
            empty_indices = tf.where(tf.equal(bucket_keys[:, 0],
                                              self.sentinel_value))
            empty_indices = tf.cast(empty_indices, tf.int32)
            store_idx = self._flatten_index(empty_indices[0, 0], bucket_index)
            return self._update_at_index(store_idx, store_key, store_vals)

    def _get_store_op_full(self, store_key, store_vals, bucket_index,
                           bucket_keys):
        """get an op to store given keys and values when there are no empty
        slots.

        Returns an op with no output that will run all of the require updates.
        """
        # TODO: what should this do? LRU? Need some accounting for that,
        # otherwise some kind of interpolation for lossily storing it in the
        # bucket?
        # for now we are just going to choose at random, which is surely a
        # terrible strategy, but at least it will run
        with tf.name_scope('store_full'):
            idx = tf.random_uniform([], minval=0, maxval=self._bucket_size,
                                    dtype=tf.int32)
            store_idx = self._flatten_index(idx, bucket_index)
            return self._update_at_index(store_idx, store_key, store_vals)

    def _get_averaged_value(self, values, similarities):
        """get a weighted sum of values."""
        weighted_values = tf.expand_dims(similarities, 1) * values
        all_values = tf.reduce_sum(weighted_values, axis=0)
        return all_values

    def get(self, key):
        """Get the values in the dictionary corresponding to a particular key,
        or zeros if the key is not present.

        The process is as follows:
            - compute hash of `key`
            - lookup all keys and values with matching hash
            - compute similarities between all matching keys and `key`
            - return average of all matching values, weighted by similarities.

        The default similarity is the cosine distance.

        Args:
            key (tensor): `[key_size]` batch of keys to look up.

        Returns:
            value (tuple): associated values.
        """
        with tf.name_scope('dnd/get'):
            bucket_keys, bucket_values, _ = self._get_bucket(key)
            # compute similarities
            similarities = self._similarity_measure(key, bucket_keys)
            # where the keys are sentinel, mask it out
            used_positions = tf.not_equal(bucket_keys[:, 0],
                                          self.sentinel_value)
            values = [tf.boolean_mask(val, used_positions)
                      for val in bucket_values]
            similarities = tf.boolean_mask(similarities, used_positions)
            results = tuple(self._get_averaged_value(val, similarities)
                            for val in values)
        return results, similarities
