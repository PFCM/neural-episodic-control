"""tests for sh.py"""
import tensorflow as tf

from tensorflow.python.platform import test

import dnd.lsh as lsh


class TestSimHash(test.TestCase):

    def test_shapes(self):
        """just ensure we get the correct shapes back"""
        inputs = tf.random_normal([10, 15, 5])

        hashed = lsh.simhash(inputs, 16)

        self.assertEqual([10, 1], hashed.get_shape().as_list())

    def test_max_bits(self):
        """not an exhaustive test"""
        inputs = tf.random_normal([10, 100])

        hashed = lsh.simhash(inputs, 16)

        in_range = tf.reduce_all(tf.less(hashed, 2**16))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(100):
                self.assertTrue(sess.run(in_range))
