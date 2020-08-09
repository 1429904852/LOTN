#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof


import numpy as np
import tensorflow as tf


def bi_dynamic_rnn(cell, inputs, n_hidden, length, scope_name):
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    return outputs


def dynamic_rnn(cell, inputs, n_hidden, length, scope_name):
    (outputs_fw, outputs_bw), state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    # outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    return outputs_fw


def lstm_cell(batch_size, vocabulary_size, num_nodes, input):
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))

    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))

    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))

    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))

    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

    input_gate = tf.sigmoid(tf.matmul(input, ix) + tf.matmul(saved_output, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(input, fx) + tf.matmul(saved_output, fm) + fb)
    update = tf.matmul(input, cx) + tf.matmul(saved_output, cm) + cb
    state = saved_state * forget_gate + tf.tanh(update) * input_gate
    output_gate = tf.sigmoid(tf.matmul(input, ox) + tf.matmul(saved_output, om) + ob)

    return output_gate * tf.tanh(state), state


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def last_mean_with_len(outputs, n_hidden, length, max_len):
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * max_len + (length - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * 2n_hidden+1
    return outputs


def softmax_layer(inputs, n_hidden, sen_len, keep_prob, l2_reg, n_class, scope_name='1'):
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_class)), np.sqrt(6.0 / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    with tf.name_scope('softmax'):
        # [batch, 80, 600]
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        # [batch*80, 600]
        outputs = tf.reshape(outputs, [-1, n_hidden])
        # [batch*20, 3]
        predict = tf.matmul(outputs, w) + b
        predict = tf.reshape(predict, [-1, sen_len, n_class])
        predict = tf.nn.softmax(predict)
    return predict


def softmax_layer1(inputs, n_hidden, keep_prob, l2_reg, n_class, scope_name='1'):
    w = tf.get_variable(
        name='softmax_w_1' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_class)), np.sqrt(6.0 / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b_1' + scope_name,
        shape=[n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    with tf.name_scope('softmax_1'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b
        predict = tf.nn.softmax(predict)
    return predict