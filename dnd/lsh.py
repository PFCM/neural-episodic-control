"""Tools for locality sensitive hashing, designed to be used for fast nearest
neighbour in the differentiable neural dictionary"""
import numpy as np
import tensorflow as tf


def simhash(inputs, num_bits):
    """SimHash the inputs into an integer with `num_bits` used bits.
    The process is:
        - flatten inputs into `[batch_size, ?]`
        - multiply by a (fixed) random gaussian matrix.
        - convert to bits with the sign function (and appropriate shifting and
          scaling)
        - convert from a `[batch_size, num_bits]` matrix to a `[batch_size, 1]`
          vector of integers.

    The standard process requires sign(0) = 1, as usual. However, most maths
    libraries define sign(0) = 0. At this stage we ignore it because it is
    highly unlikely, but if the inputs are sparse it's a possibility and should
    probably be adressed (it's not going to break anything, but it will lead to
    some potentially unexpected hashes which _might_ break the locality
    sensitive property).

    Args:
        inputs (tensor): tensor of whatever shape, with the batch on the first
            axis. Apart from the batch size, the shape does need to be defined.
        num_bits (int): number of buckets we hash to.

    Returns:
        tensor: `[batch_size, 1]` integer tensor.
    """
    with tf.variable_scope('simhash'):
        original_shape = inputs.get_shape().as_list()
        num_features = np.prod(original_shape[1:])
        inputs = tf.reshape(inputs, [-1, num_features])
        projection_matrix = tf.get_variable(
            'projection', shape=[num_features, num_bits],
            initializer=tf.random_normal_initializer())
        projected = tf.matmul(inputs, projection_matrix)
        bits = tf.sign(projected) * 0.5 + 0.5
        # return bits
        bits = tf.cast(bits, tf.int32)
        # convert to single int
        bases = 2 ** tf.range(num_bits)
        # hope for broadcasting
        index = tf.reduce_sum(bits * tf.expand_dims(bases, 0),
                              axis=1, keep_dims=True)
        return index
