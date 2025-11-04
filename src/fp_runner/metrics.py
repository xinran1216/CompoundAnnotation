
import tensorflow as tf

def tanimoto_coefficient(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred_rounded)
    sum_true = tf.reduce_sum(y_true)
    sum_pred = tf.reduce_sum(y_pred_rounded)
    union = sum_true + sum_pred - intersection
    return tf.math.divide_no_nan(intersection, union)

def sokal_sneath_3(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    a = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_rounded, 1)), tf.float32))
    b = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_rounded, 0)), tf.float32))
    c = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_rounded, 1)), tf.float32))
    d = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_rounded, 0)), tf.float32))
    return 0.25 * (tf.math.divide_no_nan(a, a+b) + tf.math.divide_no_nan(a, a+c) + tf.math.divide_no_nan(d, b+d) + tf.math.divide_no_nan(d, c+d))

CUSTOM_OBJECTS = {
    "tanimoto_coefficient": tanimoto_coefficient,
    "sokal_sneath_3": sokal_sneath_3,
}
