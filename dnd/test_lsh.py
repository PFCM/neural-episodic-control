"""tests for sh.py"""
import tensorflow as tf

from tensorflow.python.platform import test

import dnd.lsh as lsh


class TestSimHash(test.TestCase):

    def test_shapes(self):
        """just ensure we get the correct shapes back"""
        inputs = tf.random_normal([10, 15])

        conf = lsh.get_simhash_config(15, 16)
        hashed = lsh.simhash(inputs, conf)

        self.assertEqual([10, 1], hashed.get_shape().as_list())

    def test_max_bits(self):
        """not an exhaustive test"""
        inputs = tf.random_normal([10, 100])

        conf = lsh.get_simhash_config(100, 4)
        hashed = lsh.simhash(inputs, conf)

        in_range = tf.reduce_all(tf.less(hashed, 2**4))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(100):
                self.assertTrue(sess.run(in_range))

    def test_deterministic(self):
        """ensure it gives the same result twice"""
        inputs = tf.get_variable(name='inputs',
                                 shape=[1, 20])
        conf = lsh.get_simhash_config(20, 8)
        hashed = lsh.simhash(inputs, conf)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            a = sess.run(hashed)
            b = sess.run(hashed)

            self.assertEqual(a, b)

    def test_one_at_a_time(self):
        """make sure if gives a scalr back if given a vector"""
        inputs = tf.random_normal([5])
        conf = lsh.get_simhash_config(5, 2)
        hashed = lsh.simhash(inputs, conf)

        self.assertEqual(hashed.get_shape().as_list(), [])

    def test_config_reuse(self):
        """make sure it is trying to reuse variables"""
        conf = lsh.get_simhash_config(10, 10)

        with self.assertRaisesRegex(ValueError, '.* already exists'):
            conf = lsh.get_simhash_config(10, 12)

    def test_different_config_different_results(self):
        inputs = tf.get_variable(name='inputs',
                                 shape=[50, 100])

        with tf.variable_scope('a'):
            conf_a = lsh.get_simhash_config(100, 8)
        with tf.variable_scope('b'):
            conf_b = lsh.get_simhash_config(100, 8)

        hash_a = lsh.simhash(inputs, conf_a)
        hash_b = lsh.simhash(inputs, conf_b)

        equal = tf.reduce_all(tf.equal(hash_a, hash_b))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertFalse(sess.run(equal))
