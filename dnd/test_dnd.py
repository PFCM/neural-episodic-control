"""tests for the dnd"""
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

import dnd.hashdict as hashdict


def _row_in(row, mat):
    """check a row is in a matrix"""
    return np.any(np.equal(mat, row).all(1))


class TestDND(test.TestCase):

    simple_shapes = [[10]]

    def test_store_empty(self):
        """make sure we can store values in the dictionary"""
        dictionary = hashdict.HashDND(4, 10, 5, TestDND.simple_shapes)

        key = tf.get_variable('key', shape=[5])
        value = tf.get_variable('value', shape=TestDND.simple_shapes[0])

        store_op = dictionary.store(key, [value])

        # now there should be non-infs in the dicts key/value vars
        key_stored = tf.reduce_any(tf.not_equal(dictionary._keys,
                                                dictionary.sentinel_value))
        val_stored = tf.reduce_any(tf.not_equal(dictionary._values[0],
                                                dictionary.sentinel_value))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # run the store op
            sess.run(store_op)

            self.assertTrue(sess.run(key_stored))
            self.assertTrue(sess.run(val_stored))

    def test_store_full(self):
        """make sure stuff keeps going in even when it is full up."""
        # NOTE: what is supposed to happen in this case is not settled, so this
        # is likely to need to be redone
        dictionary = hashdict.HashDND(1, 1, 5, TestDND.simple_shapes)

        # just use random keys to make sure we cover everything
        key = tf.random_normal([5])
        # random values as well
        value = tf.random_normal(TestDND.simple_shapes[0])

        keys_full = tf.reduce_all(tf.not_equal(dictionary._keys,
                                               dictionary.sentinel_value))
        vals_full = tf.reduce_all(tf.not_equal(dictionary._keys,
                                               dictionary.sentinel_value))

        store_op = dictionary.store(key, [value])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # there are 2 buckets, each with a single slot
            # but there are likely to be collisions, so it may take a few to
            # fill up
            for _ in range(4):
                _, newkey, newval, allkeys, allvals = sess.run(
                    [store_op, key, value, dictionary._keys] +
                    dictionary._values)
                self.assertTrue(_row_in(newkey, allkeys))
                self.assertTrue(_row_in(newval, allvals))
            # check that it is full
            self.assertTrue(sess.run(keys_full), '{}'.format(allkeys))
            self.assertTrue(sess.run(vals_full))

            # add another one
            _, newkey, newval, allkeys, allvals = sess.run(
                [store_op, key, value, dictionary._keys] +
                dictionary._values)
            self.assertTrue(_row_in(newkey, allkeys))
            self.assertTrue(_row_in(newval, allvals))

    def test_get(self):
        """make sure we can pull things back out"""
        dictionary = hashdict.HashDND(4, 10, 5, TestDND.simple_shapes)

        key = tf.get_variable('key', shape=[5])
        value = tf.get_variable('value', shape=TestDND.simple_shapes[0])

        store_op = dictionary.store(key, [value])
        result = dictionary.get(key)[0]

        # should be pretty much perfect retrieval when nothing else is in

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # run the store op
            sess.run(store_op)

            # now pull it out
            original_val, retrieved_val = sess.run([value, result])

            self.assertNDArrayNear(original_val, retrieved_val, 1e-5)
