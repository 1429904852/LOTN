#!/usr/bin/env python
# encoding: utf-8

from transfer_module.nn_layer import softmax_layer, bi_dynamic_rnn
from transfer_module.config import *
from transfer_module.utils import load_w2v, score_BIO, batch_iter, load_inputs_10
import datetime
import numpy as np
import os


def LOTN(inputs, inputs_s_1, position, sen_len, target, sen_len_tr, attention1, keep_prob1, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    inputs_s_1 = tf.nn.dropout(inputs_s_1, keep_prob=keep_prob1)

    with tf.variable_scope("rnn"):
        hiddens_t = bi_dynamic_rnn(cell, inputs_s_1, FLAGS.n_hidden, sen_len, 'sen12')

    with tf.variable_scope("rnn1"):
        hiddens_s = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, 'sen13')

    hidden_total = tf.concat([hiddens_t, hiddens_s], 2)

    outputs = softmax_layer(hidden_total, 4 * FLAGS.n_hidden, FLAGS.max_sentence_len, keep_prob1, FLAGS.l2_reg,
                            FLAGS.n_class, 'sen22')
    outputs_att = softmax_layer(hiddens_s, 2 * FLAGS.n_hidden, FLAGS.max_sentence_len, keep_prob1, FLAGS.l2_reg,
                                FLAGS.plority, 'sen33')

    return outputs, outputs_att


def preprocess(word_id_mapping):
    tr_x, tr_sen_len, tr_target_word, tr_tar_len, tr_y, tr_position, tr_attention = load_inputs_10(
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
        y = tf.placeholder(tf.float32, [None, FLAGS.max_sentence_len, FLAGS.n_class], name='input_y')
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

    # target_1 = tf.reduce_mean(tf.nn.embedding_lookup(word_embedding, target_words), 1, keep_dims=True)
    # batch_size = tf.shape(inputs_s)[0]
    # target_2 = tf.zeros([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim]) + target_1

    inputs_s = tf.concat([inputs_s, input_position], 2)

    # inputs_s = tf.concat([inputs_s, target_2], 2)

    target = tf.nn.embedding_lookup(word_embedding, target_words)
    prob, prob1 = LOTN(inputs_s, inputs_s_1, position, sen_len, target, tar_len, attention1, keep_prob1, FLAGS.t1)

    loss1 = loss_func(y, prob)
    loss2 = loss_func(attention1, prob1)

    loss = loss1 + FLAGS.Auxiliary_loss * loss2

    # acc_num, acc_prob = acc_func(y, prob)

    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)

    true_y = tf.argmax(y, 2, name='true_y_1')
    pred_y = tf.argmax(prob, 2, name='pred_y_1')

    true_attention = tf.argmax(attention1, 2, name='true_attention_1')
    pred_attention = tf.argmax(prob1, 2, name='pred_attention_1')

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
            ckpt = tf.train.get_checkpoint_state(FLAGS.pre_trained_path)
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
            step, _, losses = sess.run([global_step, optimizer, loss], feed_dict)
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

            tf_true, tf_pred, tf_true_attention, tf_pred_attention, prob1, _loss = sess.run(
                [true_y, pred_y, true_attention, pred_attention, prob, loss], feed_dict)
            cost = 0
            pre_label, att_pre_label, true_label, att_true_label = [], [], [], []
            for logit, position1, att_logit, att_position1, length in zip(tf_pred, tf_true, tf_pred_attention,
                                                                          tf_true_attention, tr_sen_len_dev):
                logit = logit[:length]
                tr_position = position1[:length]

                att_logit = att_logit[:length]
                tr_att_position = att_position1[:length]

                cost += _loss * length
                pre_label.append(logit)
                true_label.append(tr_position)

                att_pre_label.append(att_logit)
                att_true_label.append(tr_att_position)

            return pre_label, att_pre_label, true_label, att_true_label, cost

        checkpoint_dir = os.path.abspath(FLAGS.saver_checkpoint)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        max_f1 = 0
        max_recall = 0
        max_precision = 0
        last_improvement = 0
        require_improvement_iterations = 5
        max_label, max_att_label = None, None
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
                         tr_attention_dev)), 500,
                1, False)
            label_pp, att_label_pp, label_tt, att_label_tt = [], [], [], []
            cost1 = 0
            for batch_ in batches_test:
                te_x_batch, te_sen_len_batch, te_target_batch, te_tar_len_batch, te_y_batch, te_position_batch, te_attention_batch = zip(
                    *batch_)
                label_p, att_label_p, label_t, att_label_t, _loss = dev_step(te_x_batch, te_sen_len_batch,
                                                                             te_target_batch, te_tar_len_batch,
                                                                             te_y_batch, te_position_batch,
                                                                             te_attention_batch)
                label_pp += label_p
                label_tt += label_t

                att_label_pp += att_label_p
                att_label_tt += att_label_t

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
                max_att_label = att_label_pp
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
