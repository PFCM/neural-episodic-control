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
    nearest neighbour lookup. Assumes keys are vectors."""

    sentinal_values = {tf.float32: np.inf}

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

    def store(key, value):
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
        pass

    def _get_averaged_value(bucket, values, similarities):
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

    def get(key):
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
        bucket = simhash(key, self._hash_size)
        # slice out the bucket
        neighbour_keys = self._keys[bucket, ...]
        # compute similarities
        neighbour_sims = tf.matmul(tf.nn.l2_normalize(key, dim=1),
                                   tf.nn.l2_normalize(neighbour_keys, dim=1),
                                   transpose_b=True)
        # pull out values for the bucket, weight by similarities and sum
