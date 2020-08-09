#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    # with tf.variable_scope("attention_sentence"):
    #     alpha = tf.div(inputs, _sum, name='attention_sen')
    #     # alpha = tf.Variable(alpha, name='attention_sen')
    #     # alpha = tf.Variable(alpha1, tf.float32)
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    # [None*max_len, 2*hidden]
    # [batch_size*max_sentence_len, 2*hidden+embedding+position_embedding]
    inputs = tf.reshape(inputs, [-1, n_hidden])
    # [None, max_len, 2*hidden+embedding+position_embedding]
    # inputs_tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    # [None, 2*hidden, 1]
    attend = tf.expand_dims(attend, 2)
    # [None, 1, max_len]
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta
    return outputs


def self_attention(keys, num_units, num_heads, scope='multihead_attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.nn.relu(
            tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        K = tf.nn.relu(
            tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        V = tf.nn.relu(
            tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks
        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += keys
        outputs = normalize(outputs)
    return outputs


def position_attention_layer(inputs, attend, input_position, position_embedding_dim, length, n_hidden, l2_reg,
                             random_base, layer_id=1):
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    w1 = tf.get_variable(
        name='att_p1_' + str(layer_id),
        shape=[position_embedding_dim, 1],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )

    # [None*max_len, 2*hidden]
    inputs = tf.reshape(inputs, [-1, n_hidden])
    # [batch*max_len, position_embedding]

    inputs_position = tf.reshape(input_position, [-1, position_embedding_dim])

    tmp_position = tf.reshape(tf.matmul(inputs_position, w1), [batch_size, 1, max_len])

    # [None, max_len, 2*hidden]
    inputs_tmp = tf.tanh(tf.matmul(inputs, w) + b)
    tmp = tf.reshape(inputs_tmp, [-1, max_len, n_hidden])
    # tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    # [None, max_len, 2*hidden]
    # tmp = tf.concat([tmp, tmp_position], 2)
    # tmp = tf.reshape(tmp, [-1, n_hidden + position_embedding_dim])
    # tmp = tf.reshape(tf.matmul(tmp, w2), [-1, max_len, n_hidden])

    # tmp = tf.tanh(tmp)

    # [None, 2*hidden, 1]
    attend = tf.expand_dims(attend, 2)
    # [None, 1, max_len]
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    tmp += tmp_position

    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


# [batch, max_sen_len, 2*hidden]
def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.matmul(inputs, w)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha
