#!/usr/bin/env python
# encoding: utf-8

from transfer_module.nn_layer import softmax_layer, bi_dynamic_rnn
from transfer_module.config import *
from transfer_module.utils import load_w2v, score_BIO, batch_iter, load_inputs_11
import datetime
import numpy as np
import os


def LOTN_crf(inputs, inputs_s_1, position, y, sen_len, target, sen_len_tr, attention1, keep_prob1, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    num_tags = 3

    with tf.variable_scope("rnn"):
        hiddens_t = bi_dynamic_rnn(cell, inputs_s_1, FLAGS.n_hidden, sen_len, 'sen12')

    with tf.variable_scope("rnn1"):
        hiddens_s = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, 'sen13')

    hidden_total = tf.concat([hiddens_t, hiddens_s], 2)

    outputs_att = softmax_layer(hiddens_s, 2 * FLAGS.n_hidden, FLAGS.max_sentence_len, keep_prob1, FLAGS.l2_reg, FLAGS.plority, 'sen33')

    W = tf.get_variable("projection_w", [4 * FLAGS.n_hidden, num_tags])
    b = tf.get_variable("projection_b", [num_tags])
    x_reshape = tf.reshape(hidden_total, [-1, 4 * FLAGS.n_hidden])
    projection = tf.matmul(x_reshape, W) + b
    outputs = tf.reshape(projection, [-1, FLAGS.max_sentence_len, num_tags], name='outputs_crf')
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(outputs, y, sen_len)
    return outputs, log_likelihood, transition_params, outputs_att


def preprocess(word_id_mapping):
    tr_x, tr_sen_len, tr_target_word, tr_tar_len, tr_y, tr_position, tr_attention = load_inputs_11(
        FLAGS.train_file_path,
        word_id_mapping,
        FLAGS.max_sentence_len,
        FLAGS.max_target_len
    )

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(tr_x)))

    x_shuffled = tr_x[shuffle_indices]
    tr_sen_len_shuffled = tr_sen_len[shuffle_indices]
    tr_target_word_shuffled = tr_target_word[shuffle_indices]
    tr_tar_len_shuffled = tr_tar_len[shuffle_indices]
    tr_y_shuffled = tr_y[shuffle_indices]
    tr_position_shuffled = tr_position[shuffle_indices]
    tr_attention_shuffled = tr_attention[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(tr_x)))
    tr_x_train, tr_x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    tr_sen_len_train, tr_sen_len_dev = tr_sen_len_shuffled[:dev_sample_index], tr_sen_len_shuffled[dev_sample_index:]
    tr_target_word_train, tr_target_word_dev = tr_target_word_shuffled[:dev_sample_index], tr_target_word_shuffled[
                                                                                           dev_sample_index:]
    tr_tar_len_train, tr_tar_len_dev = tr_tar_len_shuffled[:dev_sample_index], tr_tar_len_shuffled[dev_sample_index:]
    tr_y_train, tr_y_dev = tr_y_shuffled[:dev_sample_index], tr_y_shuffled[dev_sample_index:]
    tr_position_train, tr_position_dev = tr_position_shuffled[:dev_sample_index], tr_position_shuffled[
                                                                                  dev_sample_index:]

    tr_attention_train, tr_attention_dev = tr_attention_shuffled[:dev_sample_index], tr_attention_shuffled[
                                                                                     dev_sample_index:]

    print("Train/Dev split: {:d}/{:d}".format(len(tr_x_train), len(tr_x_dev)))
    return tr_x_train, tr_sen_len_train, tr_target_word_train, tr_tar_len_train, \
           tr_y_train, tr_position_train, tr_attention_train, tr_x_dev, tr_sen_len_dev, tr_target_word_dev, \
           tr_tar_len_dev, tr_y_dev, tr_position_dev, tr_attention_dev


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    tr_x_train, tr_sen_len_train, tr_target_word_train, tr_tar_len_train, \
    tr_y_train, tr_position_train, tr_attention_train, tr_x_dev, tr_sen_len_dev, tr_target_word_dev, \
    tr_tar_len_dev, tr_y_dev, tr_position_dev, tr_attention_dev = preprocess(word_id_mapping)

    keep_prob1 = tf.placeholder(tf.float32, name='input_keep_prob1')
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='input_x')
        y = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='input_y')
        sen_len = tf.placeholder(tf.int32, [None], name='input_sen_len')
        target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name='input_target')
        tar_len = tf.placeholder(tf.int32, [None], name='input_tar_len')
        position = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='position')
        attention1 = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len, FLAGS.plority],
                                    name='attention_parameter_1')
    inputs_s = tf.nn.embedding_lookup(word_embedding, x)
    inputs_s_1 = tf.nn.embedding_lookup(word_embedding, x)

    position_embeddings = tf.get_variable(
        name='position_embedding',
        shape=[FLAGS.max_sentence_len, FLAGS.position_embedding_dim],
        initializer=tf.random_uniform_initializer(-FLAGS.random_base, FLAGS.random_base),
        regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    )

    input_position = tf.nn.embedding_lookup(position_embeddings, position)
    inputs_s = tf.concat([inputs_s, input_position], 2)

    target = tf.nn.embedding_lookup(word_embedding, target_words)

    unary_scores, prob, transition_params, prob11 = LOTN_crf(inputs_s, inputs_s_1, position, y, sen_len, target, tar_len, attention1, keep_prob1, FLAGS.t1)
    loss1 = tf.reduce_mean(-prob)

    loss2 = loss_func(attention1, prob11)

    loss = loss1 + 0.1 * loss2
    # acc_num, acc_prob = acc_func(y, prob)

    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if FLAGS.pre_trained == "sentence_tranfer":
            pre_trained_variables = [v for v in tf.global_variables() if
                                     v.name.startswith("rnn/sen12") and "Adam" not in v.name]
            print(pre_trained_variables)
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state('data/amazon/checkpoint8/')
            # ckpt = tf.train.get_checkpoint_state('data/yelp/checkpoint8/')
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(i, x_f, sen_len_f, target, tl, yi, x_poisition, x_attention, kp1):
            feed_dict = {
                x: x_f,
                y: yi,
                sen_len: sen_len_f,
                target_words: target,
                tar_len: tl,
                position: x_poisition,
                attention1: x_attention,
                keep_prob1: kp1
            }
            step, _, losses, a_transition_params, m_unary_scores = sess.run([global_step, optimizer, loss,
                                                                             transition_params, unary_scores],
                                                                            feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: Iter {}, step {}, loss {:g}".format(time_str, i, step, losses))

        def dev_step(te_x_f, te_sen_len_f, te_target, te_tl, te_yi, te_x_poisition, te_x_attention):
            feed_dict = {
                x: te_x_f,
                y: te_yi,
                sen_len: te_sen_len_f,
                target_words: te_target,
                tar_len: te_tl,
                position: te_x_poisition,
                attention1: te_x_attention,
                keep_prob1: 1.0
            }
            tf_transition_params, tf_tag, _loss = sess.run([transition_params, unary_scores, loss], feed_dict)
            label_prob = []
            label_true = []
            for logit, position1, length in zip(tf_tag, tr_y_dev, tr_sen_len_dev):
                logit = logit[:length]
                tr_position = position1[:length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, tf_transition_params)
                label_prob.append(viterbi_sequence)
                label_true.append(tr_position)
            return label_prob, label_true, _loss

        checkpoint_dir = os.path.abspath(FLAGS.saver_checkpoint)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        max_f1 = 0
        max_recall = 0
        max_precision = 0

        max_label = None
        for i in range(FLAGS.n_iter):
            batches_train = batch_iter(
                list(zip(tr_x_train, tr_sen_len_train, tr_target_word_train, tr_tar_len_train, tr_y_train,
                         tr_position_train, tr_attention_train)), FLAGS.batch_size, 1, True)
            for batch in batches_train:
                x_batch, sen_len_batch, target_batch, tar_len_batch, y_batch, position_batch, attention_batch = zip(
                    *batch)
                train_step(i, x_batch, sen_len_batch, target_batch, tar_len_batch, y_batch, position_batch,
                           attention_batch,
                           FLAGS.keep_prob1)

            batches_test = batch_iter(
                list(zip(tr_x_dev, tr_sen_len_dev, tr_target_word_dev, tr_tar_len_dev, tr_y_dev, tr_position_dev,
                         tr_attention_dev)), 500, 1, False)

            label_pp, label_tt = [], []
            cost1 = 0
            for batch_ in batches_test:
                te_x_batch, te_sen_len_batch, te_target_batch, te_tar_len_batch, te_y_batch, te_position_batch, te_attention_batch = zip(
                    *batch_)
                label_p, label_t, _loss = dev_step(te_x_batch, te_sen_len_batch, te_target_batch, te_tar_len_batch,
                                                   te_y_batch, te_position_batch, te_attention_batch)
                label_pp += label_p
                label_tt += label_t
                cost1 += _loss
            print("\nEvaluation:")

            precision, recall, f1 = score_BIO(label_pp, label_tt)
            current_step = tf.train.global_step(sess, global_step)
            print("Iter {}: step {}, loss {}, precision {:g}, recall {:g}, f1 {:g}".format(
                i, current_step, cost1, precision, recall, f1))

            if f1 > max_f1:
                max_f1 = f1
                max_precision = precision
                max_recall = recall
                max_label = label_pp
                last_improvement = i
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            print("topf1 {:g}, precision {:g}, recall {:g}".format(max_f1, max_precision, max_recall))
            print("\n")
            # if i - last_improvement > require_improvement_iterations:
            #     print('No improvement found in a while, stop running')
            #     break
        fp = open(FLAGS.prob_file, 'w')
        for ws in max_label:
            fp.write(' '.join([str(w) for w in ws]) + '\n')


if __name__ == '__main__':
    tf.app.run()
