#! /usr/bin/env python
import tensorflow as tf
from transfer_module.utils import load_w2v, load_inputs_10, batch_iter, score_BIO

tf.app.flags.DEFINE_string('test_file_path', 'data/14lap/test.txt', 'testing file')
tf.app.flags.DEFINE_string('test_file_path', 'data/14res/test.txt', 'testing file')
tf.app.flags.DEFINE_string('test_file_path', 'data/15res/test.txt', 'testing file')
tf.app.flags.DEFINE_string('test_file_path', 'data/16res/test.txt', 'testing file')

tf.app.flags.DEFINE_string('embedding_file_path', 'data/laptop/laptop_2014_840b_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/14res/res14_2014_840b_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/15res/res_15_2014_840b_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/16res/res_16_2014_840b_300.txt', 'embedding file')

tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')

tf.flags.DEFINE_string("checkpoint_dir", "data/14lap/checkpoint", "Checkpoint directory")
tf.flags.DEFINE_string("checkpoint_dir", "data/14res/checkpoint", "Checkpoint directory")
tf.flags.DEFINE_string("checkpoint_dir", "data/15res/checkpoint", "Checkpoint directory")
tf.flags.DEFINE_string("checkpoint_dir", "data/16res/checkpoint", "Checkpoint directory")

tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_target_len', 10, 'max target length')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('plority', 2, 'number of distinct class')

tf.app.flags.DEFINE_string('prob_file', 'prob_14res_label_case.txt', 'prob1')
tf.app.flags.DEFINE_string('true_file', 'true_14res_label_case.txt', 'true1')
tf.app.flags.DEFINE_string('att_prob_file', 'att_prob_14res_label_case.txt', 'prob2')
tf.app.flags.DEFINE_string('att_true_file', 'att_true_14res_label_case.txt', 'true2')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

word_id_mapping, _ = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
te_x, te_sen_len, te_target_word, te_tar_len, te_y, te_position, te_attention = load_inputs_10(
    FLAGS.test_file_path,
    word_id_mapping,
    FLAGS.max_sentence_len,
    FLAGS.max_target_len)

print("\nTest...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("inputs/input_x").outputs[0]
        y = graph.get_operation_by_name("inputs/input_y").outputs[0]
        sen_len = graph.get_operation_by_name("inputs/input_sen_len").outputs[0]
        target_words = graph.get_operation_by_name("inputs/input_target").outputs[0]
        tar_len = graph.get_operation_by_name("inputs/input_tar_len").outputs[0]

        attention1 = graph.get_operation_by_name("inputs/attention_parameter_1").outputs[0]

        true_y = graph.get_operation_by_name("true_y_1").outputs[0]
        pred_y = graph.get_operation_by_name("pred_y_1").outputs[0]

        true_attention = graph.get_operation_by_name("true_attention_1").outputs[0]
        pred_attention = graph.get_operation_by_name("pred_attention_1").outputs[0]

        position = graph.get_operation_by_name("inputs/position").outputs[0]
        keep_prob1 = graph.get_operation_by_name("input_keep_prob1").outputs[0]

        # transition_params = graph.get_operation_by_name("transitions").outputs[0]
        # unary_scores = graph.get_operation_by_name("outputs_crf").outputs[0]


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

            tf_true, tf_pred, tf_true_attention, tf_pred_attention = sess.run([true_y, pred_y, true_attention, pred_attention], feed_dict)

            pre_label, att_pre_label, true_label, att_true_label = [], [], [], []
            for logit, position1, att_logit, att_position1, length in zip(tf_pred, tf_true, tf_pred_attention, tf_true_attention, te_sen_len):
                logit = logit[:length]
                tr_position = position1[:length]

                att_logit = att_logit[:length]
                tr_att_position = att_position1[:length]

                pre_label.append(logit)
                true_label.append(tr_position)

                att_pre_label.append(att_logit)
                att_true_label.append(tr_att_position)

            return pre_label, att_pre_label, true_label, att_true_label

        batches_test = batch_iter(
            list(zip(te_x, te_sen_len, te_target_word, te_tar_len, te_y, te_position, te_attention)), 2000, 1, False)

        label_pp = []
        label_pp_2 = []
        att_label_pp = []
        att_label_tt = []

        label_tt = []
        for batch_ in batches_test:
            te_x_batch, te_sen_len_batch, te_target_batch, te_tar_len_batch, te_y_batch, te_position_batch, te_attention_batch = zip(
                *batch_)
            label_p, att_label_p, label_t, att_label_t = dev_step(te_x_batch, te_sen_len_batch, te_target_batch, te_tar_len_batch, te_y_batch,
                                          te_position_batch, te_attention_batch)
            label_pp += label_p
            label_tt += label_t

            att_label_pp += att_label_p
            att_label_tt += att_label_t

        precision, recall, f1 = score_BIO(label_pp, label_tt)
        print("topf1 {:g}, precision {:g}, recall {:g}".format(f1, precision, recall))

        fp = open(FLAGS.prob_file, 'w')
        for ws in label_pp:
            fp.write(' '.join([str(w) for w in ws]) + '\n')

        fp = open(FLAGS.true_file, 'w')
        for ws in label_tt:
            fp.write(' '.join([str(w) for w in ws]) + '\n')

        fp = open(FLAGS.att_prob_file, 'w')
        for ws in att_label_pp:
            fp.write(' '.join([str(w) for w in ws]) + '\n')

        fp = open(FLAGS.att_true_file, 'w')
        for ws in att_label_tt:
            fp.write(' '.join([str(w) for w in ws]) + '\n')
