# -*- coding:utf-8 -*-

import numpy as np
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_text(filename):
    with open(filename,'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_keywords(text,number):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=5)
    print(u'关键词：')
    for item in tr4w.get_keywords(number, word_min_len=1):#返回的是一个字典  {'weight': xxx , 'word': xxx}
        print(item['word'], item['weight'])

def ks_plot(text, number):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='no_stop_words')
    data = pd.DataFrame(data = tr4s.key_sentences)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(data['weight'], 'ro-', lw=2, ms=5, alpha=0.7)
    plt.grid(b=True)
    plt.xlabel(u'句子', fontsize=14)
    plt.ylabel(u'重要度', fontsize=14)
    plt.title(u'句子的重要度曲线', fontsize=18)
    plt.show()

    key_sentences = tr4s.get_key_sentences(num= number, sentence_min_len=4)
    for sentence in key_sentences:
        print(sentence['weight'], sentence['sentence'])

if __name__ == '__main__':
    print('读取数据.')
    filename = 'novel.txt'
    text = get_text(filename)

    print('提取关键字.')
    number = 10 #前十个关键字
    get_keywords(text, number)

    print('关键句子曲线与前{}关键句:'.format(number))
    ks_plot(text,number)


