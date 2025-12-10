import tensorflow as tf


def tanimoto_coefficient(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    inter = tf.reduce_sum(y_true * y_pred_rounded)
    sum_true = tf.reduce_sum(y_true)
    sum_pred = tf.reduce_sum(y_pred_rounded)
    union = sum_true + sum_pred - inter
    return tf.math.divide_no_nan(inter, union)


def sokal_sneath_3(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)

    y_true_bool = tf.cast(tf.equal(y_true, 1), tf.bool)
    y_pred_bool = tf.cast(tf.equal(y_pred_rounded, 1), tf.bool)

    a = tf.reduce_sum(tf.cast(tf.logical_and(y_true_bool, y_pred_bool), tf.float32))
    b = tf.reduce_sum(tf.cast(tf.logical_and(y_true_bool, tf.logical_not(y_pred_bool)), tf.float32))
    c = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_bool), y_pred_bool), tf.float32))
    d = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.logical_not(y_true_bool), tf.logical_not(y_pred_bool)), tf.float32)
    )

    term1 = tf.math.divide_no_nan(a, a + b)
    term2 = tf.math.divide_no_nan(a, a + c)
    term3 = tf.math.divide_no_nan(d, b + d)
    term4 = tf.math.divide_no_nan(d, c + d)

    return 0.25 * (term1 + term2 + term3 + term4)


CUSTOM_OBJECTS = {
    "tanimoto_coefficient": tanimoto_coefficient,
    "sokal_sneath_3": sokal_sneath_3,
}
