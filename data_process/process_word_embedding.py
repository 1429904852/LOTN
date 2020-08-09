#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_file_path', 'data/laptop/laptop_2014_lstm_train.txt', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'data/laptop/laptop_2014_lstm_test.txt', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'data/laptop/laptop_2014_lstm_test.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/embedding/glove.840B.300d.w2v.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_path', 'data/embedding/laptop/laptop_2014_840b_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_file', 'data/laptop_word/laptop_word.txt', 'word file')


def __read_text__(fnames):
    text = ''
    for fname in fnames:
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            _, _, text_right = [s.lower().strip() for s in lines[i].partition("\t")]
            print(text_right)
            text += text_right + " "
    return text


def fit_on_text(text):
    words = text.split()
    word2idx = dict()
    idx = 0
    for word in words:
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1
    if not os.path.exists(FLAGS.word_file):
        with open(FLAGS.word_file, 'w') as s_vocab:
            for key, value in word2idx.items():
                s_vocab.write(key + '\n')
    return word2idx


def load_word_vec(path, word2idx):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if not os.path.exists(FLAGS.embedding_path):
        with open(FLAGS.embedding_path, 'w') as s_vocab:
            for line in fin:
                tokens = line.rstrip().split()
                if tokens[0] in word2idx.keys():
                    s_vocab.write(line)


if __name__ == '__main__':
    text = __read_text__([FLAGS.train_file_path, FLAGS.test_file_path])
    word_id_mapping = fit_on_text(text.lower())
    load_word_vec(FLAGS.embedding_file_path, word_id_mapping)
