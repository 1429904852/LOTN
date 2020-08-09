#! /usr/bin/env python

import tensorflow as tf
from tranfer_attention.utils import load_w2v, batch_index, load_inputs_test

# tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/twitter_train_sentence.txt', 'testing file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/twitter/twitter_sentence.txt', 'testing file')

# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/restaurant_train_sentence.txt', 'testing file')
# tf.app.flags.DEFINE_string('test_file_path', 'data/restaurant/rest_sentence.txt', 'testing file')

# tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_train_sentence.txt', 'testing file')
tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_sentence.txt', 'testing file')

tf.app.flags.DEFINE_string('embedding_file_path', 'data/amazon/amazon_2014_840b_300.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')

# tf.flags.DEFINE_string("checkpoint_dir", "data/amazon/checkpoint8/", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "data/amazon/checkpoint20/", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "data/amazon/checkpoint40/", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("checkpoint_dir", "data/amazon/checkpoint60/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", "data/amazon/checkpoint80/", "Checkpoint directory from training run")

tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')

# tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_twitter_train_result/80_multitop_att_train_twitter1.txt', 'prob1')
# tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_twitter_train_result/80_multitop_att_train_twitter2.txt', 'prob2')
# tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_twitter_train_result/80_multitop_att_train_twitter3.txt', 'prob3')

# tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_twitter_test_result/80_multitop_att_test_twitter1.txt', 'prob1')
# tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_twitter_test_result/80_multitop_att_test_twitter2.txt', 'prob2')
# tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_twitter_test_result/80_multitop_att_test_twitter3.txt', 'prob3')

# tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_rest_train_result/80_multitop_att_train_rest1.txt', 'prob1')
# tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_rest_train_result/80_multitop_att_train_rest2.txt', 'prob2')
# tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_rest_train_result/80_multitop_att_train_rest3.txt', 'prob3')

# tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_rest_test_result/80_multitop_att_test_rest1.txt', 'prob1')
# tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_rest_test_result/80_multitop_att_test_rest2.txt', 'prob2')
# tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_rest_test_result/80_multitop_att_test_rest3.txt', 'prob3')

# tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_laptop_train_result/80_multitop_att_train_laptop1.txt', 'prob1')
# tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_laptop_train_result/80_multitop_att_train_laptop2.txt', 'prob2')
# tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_laptop_train_result/80_multitop_att_train_laptop3.txt', 'prob3')

tf.app.flags.DEFINE_string('att_s_file_1', '80_multitop_laptop_test_result/80_multitop_att_test_laptop1.txt', 'prob1')
tf.app.flags.DEFINE_string('att_s_file_2', '80_multitop_laptop_test_result/80_multitop_att_test_laptop2.txt', 'prob2')
tf.app.flags.DEFINE_string('att_s_file_3', '80_multitop_laptop_test_result/80_multitop_att_test_laptop3.txt', 'prob3')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

word_id_mapping, _ = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
te_x, te_sen_len = load_inputs_test(FLAGS.test_file_path, word_id_mapping, FLAGS.max_sentence_len)

print("\nTest...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(tf.get_default_graph().as_graph_def())
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("inputs/input_x").outputs[0]
        sen_len = graph.get_operation_by_name("inputs/input_sen_len").outputs[0]

        keep_prob1 = graph.get_operation_by_name("input_keep_prob1").outputs[0]
        keep_prob2 = graph.get_operation_by_name("input_keep_prob2").outputs[0]
        att_s_1 = graph.get_operation_by_name("attention_sentence/attention_sensen1").outputs[0]
        att_s_2 = graph.get_operation_by_name("attention_sentence_1/attention_sensen2").outputs[0]
        att_s_3 = graph.get_operation_by_name("attention_sentence_2/attention_sensen3").outputs[0]

        pred_y = graph.get_operation_by_name("pred_y_1").outputs[0]


        def get_batch_data(x_f, sen_len_f, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(x_f), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    sen_len: sen_len_f[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)


        s1, s2, s3, py = [], [], [], []
        for test1, num in get_batch_data(te_x, te_sen_len, 200, 1.0, 1.0, False):
            _py, _s1, _s2, _s3 = sess.run([pred_y, att_s_1, att_s_2, att_s_3], feed_dict=test1)
            s1 += list(_s1)
            s2 += list(_s2)
            s3 += list(_s3)
            py += list(_py)
        fp = open(FLAGS.att_s_file_1, 'w')
        for ws in s1:
            fp.write(' '.join([str(w) for w in ws[0] if w != 0.0]) + '\n')
  
        fp = open(FLAGS.att_s_file_2, 'w')
        for ws in s2:
            fp.write(' '.join([str(w) for w in ws[0] if w != 0.0]) + '\n')

        fp = open(FLAGS.att_s_file_3, 'w')
        for ws in s3:
            fp.write(' '.join([str(w) for w in ws[0] if w != 0.0]) + '\n')
