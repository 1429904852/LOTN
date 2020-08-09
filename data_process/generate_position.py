#!/usr/bin/env python
# encoding: utf-8
# @author: zhaof
import sys

lines = open(sys.argv[1]).readlines()
fp = open(sys.argv[2], 'w')
for i in range(0, len(lines), 2):
    aspect, sentence = lines[i].strip(), lines[i + 1].strip()
    words = sentence.split()
    print(words)
    ind = words.index('$T$')
    tmp = []
    position = []
    for i, word in enumerate(words[:ind], 0):
        tmp.append(word)
        position.append(str(ind - i))
    term = aspect.split()
    for seg in range(len(term)):
        tmp.append(term[seg])
        position.append(str(0))
    for i, word in enumerate(words[ind + 1:], 1):
        tmp.append(word)
        position.append(str(i))
    sentence = ' '.join(tmp)
    pos = ' '.join(position)
    # fp.write(aspect + '\n')
    # fp.write(sentence + '\n')
    fp.write(pos + '\n')
