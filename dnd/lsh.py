"""Tools for locality sensitive hashing, designed to be used for fast nearest
neighbour in the differentiable neural dictionary"""
import numpy as np
import tensorflow as tf


def get_simhash_config(input_size, hash_bits):
    """Gets any necessary configuration and data structures necessary for
    consistent hashing.

    For simhash, this just corresponds to the random matrix used to project the
    input down, but we also store some values used in the conversion from
    binary to integers.

    This function should be run once and the result stored and passed in to all
    subsequent calls to `simhash`, so that we use the same matrix every time.

    Args:
        input_size (int): size of the inputs we are going to hash.
        hash_bits (int): the number of bits we output.

    Returns:
        dict: dictionary with two keys: "matrix" corresponding to a variable
            used for the random projection and "bases" used in the conversion
            to integers.
    """
    with tf.variable_scope('simhash_config'):
        mat = tf.get_variable(
            'projection_matrix',
            shape=[input_size, hash_bits],
            initializer=tf.random_normal_initializer())
        bases = tf.expand_dims(2 ** tf.range(hash_bits), 0)
        return {'matrix': mat,
                'bases': bases}


def simhash(inputs, config):
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
        config (dict): the result of `get_simhash_config`.

    Returns:
        tensor: `[batch_size, 1]` integer tensor.
    """
    with tf.variable_scope('simhash'):
        projected = tf.matmul(inputs, config['matrix'])
        bits = tf.sign(projected) * 0.5 + 0.5
        # return bits
        bits = tf.cast(bits, tf.int32)
        # convert to single int
        index = tf.reduce_sum(bits * config['bases'],
                              axis=1, keep_dims=True)
        return index
