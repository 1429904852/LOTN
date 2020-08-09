from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from ABSA.config import *
import numpy as np

_EPSILON = 10 ** -4


class CustomLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self,
                 num_units=300,
                 forget_bias=0.0,
                 reuse=None,
                 layer_norm: bool = False,
                 norm_shift: float = 0.0,
                 norm_gain: float = 1.0,  # layer normalization
                 dropout_keep_prob_in: float = 1.0,
                 dropout_keep_prob_h: float = 1.0,
                 dropout_keep_prob_out: float = 1.0,
                 dropout_keep_prob_gate: float = 1.0,
                 dropout_keep_prob_forget: float = 1.0,
                 dropout_prob_seed: int = None,
                 variational_dropout: bool = False,
                 recurrent_dropout: bool = False
                 ):
        super(CustomLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = math_ops.tanh

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

        self._recurrent_dropout = recurrent_dropout
        self._variational_dropout = variational_dropout

        self._seed = dropout_prob_seed
        self._keep_prob_i = dropout_keep_prob_in
        self._keep_prob_g = dropout_keep_prob_gate
        self._keep_prob_f = dropout_keep_prob_forget
        self._keep_prob_o = dropout_keep_prob_out
        self._keep_prob_h = dropout_keep_prob_h

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _layer_normalization(self, inputs, scope=None):
        shape = inputs.get_shape()[-1:]
        with vs.variable_scope(scope or "layer_norm"):
            # Initialize beta and gamma for use by layer_norm.
            g = vs.get_variable("gain", shape=shape, initializer=init_ops.constant_initializer(self._g))  # (shape,)
            s = vs.get_variable("shift", shape=shape, initializer=init_ops.constant_initializer(self._b))  # (shape,)
        m, v = nn_impl.moments(inputs, [1], keep_dims=True)  # (batch,)
        normalized_input = (inputs - m) / math_ops.sqrt(v + _EPSILON)  # (batch, shape)

        return normalized_input * g + s

    @staticmethod
    def _linear(x, weight_shape, bias=True, scope=None):
        """ linear projection (weight_shape: input size, output size) """
        with vs.variable_scope(scope or "linear"):
            w = vs.get_variable("kernel", shape=weight_shape)
            x = math_ops.matmul(x, w)
            if bias:
                b = vs.get_variable("bias", initializer=[0.0] * weight_shape[-1])
                return nn_ops.bias_add(x, b)
            else:
                return x

    def call(self, inputs, state):
        c, t = state
        args_m = array_ops.concat([inputs, t], 1)

        concat = self._linear(args_m, [args_m.get_shape()[-1], 4 * self._num_units])
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        if self._layer_norm:
            i = self._layer_normalization(i, "layer_norm_i")
            j = self._layer_normalization(j, "layer_norm_j")
            f = self._layer_normalization(f, "layer_norm_f")
            o = self._layer_normalization(o, "layer_norm_o")
        g = self._activation(j)  # gating

        # variational dropout
        if self._variational_dropout:
            i = nn_ops.dropout(i, self._keep_prob_i, seed=self._seed)
            g = nn_ops.dropout(g, self._keep_prob_g, seed=self._seed)
            f = nn_ops.dropout(f, self._keep_prob_f, seed=self._seed)
            o = nn_ops.dropout(o, self._keep_prob_o, seed=self._seed)

        gated_in = math_ops.sigmoid(i) * g

        memory = c * math_ops.sigmoid(f + self._forget_bias)

        # recurrent dropout
        if self._recurrent_dropout:
            gated_in = nn_ops.dropout(gated_in, self._keep_prob_h, seed=self._seed)

        new_c = memory + gated_in

        new_h = self._activation(new_c) * math_ops.sigmoid(o)
        with tf.variable_scope("rnn12"):
            ww = tf.get_variable(
                name='softmax_k',
                shape=[FLAGS.n_hidden, FLAGS.n_hidden],
                initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (FLAGS.n_hidden + FLAGS.n_class))),
                regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
            )
            bb = tf.get_variable(
                name='softmax_m',
                shape=[FLAGS.n_hidden],
                initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (FLAGS.n_class))),
                regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
            )

        x = math_ops.matmul(new_h, ww)
        new_t = nn_ops.bias_add(x, bb)

        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_t)

        return new_t, new_state