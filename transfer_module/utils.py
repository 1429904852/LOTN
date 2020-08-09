#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof


import numpy as np


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    # 需要循环多少个周期，每一个迭代周期需要运行多少次batch
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 每个周期运算多少个batch
        for batch_num in range(num_batches_per_epoch):
            # 开始位置
            start_index = batch_num * batch_size
            # 结束位置
            end_index = min((batch_num + 1) * batch_size, data_size)
            # 生成器生成不同的batch
            yield shuffled_data[start_index:end_index]


def batch_iter1(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    # 需要循环多少个周期，每一个迭代周期需要运行多少次batch
    num_batches_per_epoch = int(len(data) / batch_size) + (1 if len(data) % batch_size else 0)
    # num_batches_per_epoch = int((len(data)-1)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 每个周期运算多少个batch
        for batch_num in range(num_batches_per_epoch):
            # 开始位置
            start_index = batch_num * batch_size
            # 结束位置
            end_index = min((batch_num + 1) * batch_size, data_size)
            # 生成器生成不同的batch
            yield shuffled_data[start_index:end_index]


def score_BIO(predicted, golden):
    # B:0, I:1, O:2
    assert len(predicted) == len(golden)
    sum_all = 0
    sum_correct = 0
    golden_01_count = 0
    predict_01_count = 0
    correct_01_count = 0
    # print(predicted)
    # print(golden)
    for i in range(len(golden)):
        length = len(golden[i])
        # print(length)
        # print(predicted[i])
        # print(golden[i])
        golden_01 = 0
        correct_01 = 0
        predict_01 = 0
        predict_items = []
        golden_items = []
        golden_seq = []
        predict_seq = []
        for j in range(length):
            if golden[i][j] == 1:
                if len(golden_seq) > 0:  # 00
                    golden_items.append(golden_seq)
                    golden_seq = []
                golden_seq.append(j)
            elif golden[i][j] == 2:
                if len(golden_seq) > 0:
                    golden_seq.append(j)
            elif golden[i][j] == 0:
                if len(golden_seq) > 0:
                    golden_items.append(golden_seq)
                    golden_seq = []
            if predicted[i][j] == 1:
                if len(predict_seq) > 0:  # 00
                    predict_items.append(predict_seq)
                    predict_seq = []
                predict_seq.append(j)
            elif predicted[i][j] == 2:
                if len(predict_seq) > 0:
                    predict_seq.append(j)
            elif predicted[i][j] == 0:
                if len(predict_seq) > 0:
                    predict_items.append(predict_seq)
                    predict_seq = []
        if len(golden_seq) > 0:
            golden_items.append(golden_seq)
        if len(predict_seq) > 0:
            predict_items.append(predict_seq)
        golden_01 = len(golden_items)
        predict_01 = len(predict_items)
        correct_01 = sum([item in golden_items for item in predict_items])
        # print(correct_01)
        # print([item in golden_items for item in predict_items])
        # print(golden_items)
        # print(predict_items)

        golden_01_count += golden_01
        predict_01_count += predict_01
        correct_01_count += correct_01
    precision = correct_01_count/predict_01_count if predict_01_count > 0 else 0
    recall = correct_01_count/golden_01_count if golden_01_count > 0 else 0
    f1 = 2*precision*recall/(precision +recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.encode(encoding, 'ignore').decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    # a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for line in fp:
        line = line.encode('utf8', 'ignore').decode('utf8', 'ignore').split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print(u'a bad word embedding: {}'.format(line[0]))
            continue
        cnt += 1
        # if line[0] ==
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def change_y_to_onehot(y):
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": 0,
        "b": 1,
        "i": 2
    }

    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))
        label.append(label_ + [0] * (sentence_len - len(label_)))

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position)


def load_inputs_11(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')

    tag2label = {
        "o": 0,
        "b": 1,
        "i": 2
    }

    ploritylabel = {
        0: [1, 0],
        1: [0, 1]
    }
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    attention_sum_1 = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 4):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        # attention
        attention1 = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention1.append(int(attention_1))
        attention1 = attention1[:sentence_len]
        attention2 = attention1 + [0] * (sentence_len - len(attention1))
        attention3 = [ploritylabel[tag] for tag in attention2]
        attention_sum_1.append(attention3)

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(attention_sum_1)


def load_inputs_1(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": [1, 0, 0],
        "b": [0, 1, 0],
        "i": [0, 0, 1]
    }
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position)


def load_inputs_3(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": [1, 0, 0],
        "b": [0, 1, 0],
        "i": [0, 0, 1]
    }
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    attention_sum_1 = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 4):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        # attention_1
        attention1 = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention1.append(int(attention_1))
        attention1 = attention1[:sentence_len]
        attention_sum_1.append(attention1 + [0] * (sentence_len - len(attention1)))

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(attention_sum_1)


def load_inputs_10(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": [1, 0, 0],
        "b": [0, 1, 0],
        "i": [0, 0, 1]
    }
    ploritylabel = {
        0: [1, 0],
        1: [0, 1]
    }
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    attention_sum_1 = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 4):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        # attention
        attention1 = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention1.append(int(attention_1))
        attention1 = attention1[:sentence_len]
        attention2 = attention1 + [0] * (sentence_len - len(attention1))
        attention3 = [ploritylabel[tag] for tag in attention2]
        attention_sum_1.append(attention3)

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(attention_sum_1)


def load_inputs_2(input_file, word_id_file, sentence_len, target_len, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": [0, 0, 1],
        "b": [0, 1, 0],
        "i": [1, 0, 0]
    }

    parity_label = {
        "1": [0, 0, 1],
        "0": [0, 1, 0],
        "-1": [1, 0, 0]
    }

    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    pori = []

    lines = open(input_file).readlines()

    for i in range(0, len(lines), 4):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # y
        pori.append(lines[i + 3].strip())

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    pority = [parity_label[w] for w in pori]

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(pority)


def load_inputs_5(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": 0,
        "b": 1,
        "i": 2
    }
    x, y, sen_len = [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    attention_sum_1 = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 4):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]

        label_ = [tag2label[tag] for tag in pp]
        x.append(words + [0] * (sentence_len - len(words)))
        label.append(label_ + [0] * (sentence_len - len(label_)))

        # yy = pp + ["o"] * (sentence_len - len(pp))
        # yy = [tag2label[tag] for tag in yy]
        # label.append(yy)

        # attention_1
        attention1 = []
        attention_words_1 = lines[i + 3].encode(encoding).decode(encoding).lower().split()
        for attention_1 in attention_words_1:
            attention1.append(attention_1)
        attention1 = attention1[:sentence_len]
        attention_sum_1.append(attention1 + [0] * (sentence_len - len(attention1)))

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(attention_sum_1)


def load_inputs_6(input_file, word_id_file, sentence_len, target_len=10, encoding='utf8'):
    word_to_id = word_id_file
    print('load word-to-id done!')
    tag2label = {
        "o": [1, 0, 0],
        "b": [0, 1, 0],
        "i": [0, 0, 1]
    }
    x, reverse_x, y, sen_len = [], [], [], []
    target_words = []
    tar_len = []
    label = []
    position = []
    new_position = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words_2 = lines[i].encode(encoding).decode(encoding).lower().split()
        target_word = []
        for w in words_2:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        words_1 = lines[i + 1].encode(encoding).decode(encoding).lower().split()

        words, pp = [], []
        for word in words_1:
            t = word.split('\\')
            ind = t[-1]
            word = ''.join(t[:-1])
            if word in word_to_id:
                words.append(word_to_id[word])
                pp.append(ind)
        sen_len.append(len(words))
        words = words[:sentence_len]
        pp = pp[:sentence_len]
        # label_ = [tag2label[tag] for tag in pp]
        new_words = words[::-1]

        x.append(words + [0] * (sentence_len - len(words)))
        reverse_x.append(new_words + [0] * (sentence_len - len(new_words)))

        yy = pp + ["o"] * (sentence_len - len(pp))
        yy = [tag2label[tag] for tag in yy]
        label.append(yy)

        pos = []
        position_1 = lines[i + 2].encode(encoding).decode(encoding).split()
        for i in position_1:
            pos.append(int(i))
        pos = pos[:sentence_len]
        position.append(pos + [0] * (sentence_len - len(pos)))

        new_pos = pos[::-1]
        new_position.append(new_pos + [0] * (sentence_len - len(new_pos)))

    return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
           np.asarray(tar_len), np.asarray(label), np.asarray(position), np.asarray(reverse_x), np.asarray(new_position)