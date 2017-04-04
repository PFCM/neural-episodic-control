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

from dnd.lsh import simhash


class HashDND(object):
    """differentiable neural dictionary, using LSH for approximate
    nearest neighbour lookup. Assumes keys are vectors. Also assumes we only
    use float32, for now."""

    sentinal_value = np.inf

    @classmethod
    def _setup_values(cls, hash_bits, max_neighbours, value_shapes):
        """setup variables with appropriate initializers given the shapes"""
        values = []
        init = tf.constant_initializer(cls.sentinel_value)
        for i, shape in enumerate(value_shapes):
            var_shape = [2**hash_bits, max_neighbours] + shape
            var = tf.get_variable(name='value_{}'.format(i),
                                  shape=var_shape,
                                  initializer=init)
            values.append(var)
        return values


    def __init__(self, hash_bits, max_neighbours, key_size, value_shapes):
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
            value_shapes ()
        """
        self._hash_size = hash_bits
        self._keys = tf.zeros([2**hash_bits, max_neighbours, key_size])
        self._values = HashDND._setup_values(hash_bits,
                                             max_neighbours,
                                             value_shapes)
        self._hash_config = get_simhash_config(self.key_size,
                                               self._hash_size)

    def _get_bucket(self, key):
        """look up the keys and values in a given bucket correspondng to the
        hash of the given key"""
        bucket_idx = simhash(key, self._hash_config)
        keys = self._keys[bucket_idx, ...]
        values = [val[bucket_idx, ...] for val in self._values]
        return keys, values

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
            key (tensor): `[batch_size, key_size]` batch of keys
            value (tensor or list of tensors): `[batch_size, ???]` tensors to
                be stored.

        Returns:
            op: an op which carries out the above steps.
        """
        with tf.name_scope('dnd/store'):
            bucket_keys, bucket_values = self._get_bucket(key)
            # is there space?
            bucket_full = tf.reduce_all(
                tf.stack(
                    [tf.reduce_all(tf.not_equal(var, self.sentinel_value))
                     for var in bucket_values]))
            # if there is, find it and assign to it
            # if there is not, do something else??

    def _get_averaged_value(self, bucket, values, similarities):
        """get the values from a specific bucket, weighted by similarities and
        summed.

        Steps:
            - pull out the values corresponding to the (integer) bucket.
            - weight by the similarities
                - we assume these are already zeros for empty slots in the
                  bucket
            - sum
        """
        # values are [buckets, max_values, ...]
        bucket_values = tf.expand_dims(values[bucket, ...], 0)
        # similarities are [batch, max_values]
        weighted_values = similarities * bucket_values
        return tf.reduce_sum(weighted_values, axis=1)

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
            key (tensor): `[batch_size, key_size]` batch of keys to look up.

        Returns:
            value (list): list of associated values.
        """
        with tf.name_scope('dnd/get'):
            bucket_keys, bucket_values = self._get_bucket(key)
            # compute similarities
            # TODO: need to pass in a method for this?
            # pull out values for the bucket, weight by similarities and sum
