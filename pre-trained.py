#!/usr/bin/env python
# encoding: utf-8

from sklearn.metrics import precision_score, recall_score, f1_score

from pretrain_module.nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from pretrain_module.att_layer import bilinear_attention_layer
from pretrain_module.config import *
from pretrain_module.utils import load_w2v, batch_index, load_inputs
import os


def pre_train_model(inputs, sen_len, keep_prob1, keep_prob2, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)

    with tf.variable_scope("rnn"):
        hiddens_s = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sen12' + _id, 'all')
    pool_t_1 = reduce_mean_with_len(hiddens_s, sen_len)

    att_s_1 = bilinear_attention_layer(hiddens_s, pool_t_1, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sen1')
    outputs_1 = tf.squeeze(tf.matmul(att_s_1, hiddens_s))
    pool_t_2 = outputs_1 + pool_t_1

    prob = softmax_layer(pool_t_2, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob, att_s_1


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)

    word_embedding = tf.constant(w2v, name='word_embedding')

    keep_prob1 = tf.placeholder(tf.float32, name='input_keep_prob1')
    keep_prob2 = tf.placeholder(tf.float32, name='input_keep_prob2')

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='input_x')
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='input_y')
        sen_len = tf.placeholder(tf.int32, None, name='input_sen_len')

    # [batch, max_len, embedding]
    inputs_s = tf.nn.embedding_lookup(word_embedding, x)

    prob, att_1_s = pre_train_model(inputs_s, sen_len, keep_prob1, keep_prob2, FLAGS.t1)

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9).minimize(loss,
                                                                                                     global_step=global_step)
    # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
    true_y = tf.argmax(y, 1, name='true_y_1')
    pred_y = tf.argmax(prob, 1, name='pred_y_1')

    title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
        FLAGS.keep_prob1,
        FLAGS.keep_prob2,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.l2_reg,
        FLAGS.max_sentence_len,
        FLAGS.embedding_dim,
        FLAGS.n_hidden,
        FLAGS.n_class
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_sen_len, tr_y = load_inputs(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
        )
        te_x, te_sen_len, te_y = load_inputs(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
        )

        def get_batch_data(x_f, sen_len_f, yi, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        checkpoint_dir = os.path.abspath(FLAGS.saver_checkpoint)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        max_acc = 0.
        max_s_1, max_s_2, max_s_3 = None, None, None
        max_ty, max_py = None, None
        max_prob = None
        step = None
        for i in range(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, FLAGS.batch_size, FLAGS.keep_prob1,
                                           FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            s1, s2, s3, ty, py, p = [], [], [], [], [], []
            for test, num in get_batch_data(te_x, te_sen_len, te_y, 2000, 1.0, 1.0, False):
                _loss, _acc, _s1, _ty, _py, _p = sess.run(
                    [loss, acc_num, att_1_s, true_y, pred_y, prob], feed_dict=test)
                s1 += list(_s1)
                ty += list(_ty)
                py += list(_py)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            acc = acc / cnt
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc))
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            current_step = tf.train.global_step(sess, global_step)
            if acc > max_acc:
                max_acc = acc
                max_s_1 = s1
                max_ty = ty
                max_py = py
                max_prob = p
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')

        fp = open(FLAGS.att_s_file_1, 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_s_1):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0] if w != 0.0]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

if __name__ == '__main__':
    tf.app.run()