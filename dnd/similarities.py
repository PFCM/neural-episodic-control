"""Contains some similarity functions for use with the dnd."""
import tensorflow as tf


def cosine_similarity(query, bucket):
    """Cosine similarity: the cosine of the angle between two vectors.
    Also the dot product, if the vectors are normalised in the l2 norm,
    which is how it is implemented here."""
    query = tf.expand_dims(query, 1)
    query = tf.nn.l2_normalize(query, dim=0)
    bucket = tf.nn.l2_normalize(bucket, dim=1)
    return tf.squeeze(tf.matmul(bucket, query), 1, name='cos_sim')
