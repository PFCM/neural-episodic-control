"""The differentiable neural dictionary. Or at least, something inspired
by it."""
import tensorflow as tf

from lsh import simhash


class HashDND(object):
    """differentiable neural dictionary, using LSH for approximate
    nearest neighbour lookup. Assumes keys are vectors."""

    # need some way of specifying type/shape of members
    def __init__(self, hash_bits, max_neighbours, key_size):
        self._hash_size = hash_bits
        self._keys = tf.zeros([2**hash_bits, max_neighbours, key_size])

    def store(key, value):
        pass

    def get(key):
        bucket = simhash(key, self._hash_size)
        # slice out the bucket
        neighbour_keys = self._keys[bucket, ...]
        # compute similarities
        neighbour_sims = tf.matmul(tf.nn.l2_normalize(key, dim=1),
                                   tf.nn.l2_normalize(neighbour_keys, dim=1),
                                   transpose_b=True)
        # pull out values for the bucket, weight by similarities and sum
