"""Tests for similarity functions"""
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

import dnd.similarities as sim


class TestCosineSimilarity(test.TestCase):

    def test_identical(self):

        var = tf.get_variable('test', shape=[100])

        similarity = sim.cosine_similarity(var, tf.expand_dims(var, 0))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            value = sess.run(similarity)[0]  # (1,) anyway
            self.assertNear(value, 1.0, 1e-6)

    def test_orthogonal(self):
        a = tf.constant([1, 0, 0, 0], dtype=tf.float32)
        b = tf.constant([[0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=tf.float32)

        similarity = sim.cosine_similarity(a, b)

        with self.test_session() as sess:
            value = sess.run(similarity)

            self.assertNDArrayNear(value, np.zeros_like(value), 1e-6)

    def test_normalizes(self):
        var = tf.get_variable('test', shape=[100])

        similarity = sim.cosine_similarity(var, tf.expand_dims(var, 0) * 10.0)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            value = sess.run(similarity)[0]  # (1,) anyway
            self.assertNear(value, 1.0, 1e-6)
