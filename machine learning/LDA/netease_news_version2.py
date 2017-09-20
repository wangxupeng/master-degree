# -*- coding:utf-8 -*-
import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time

def stop_words():
    with open('stopword.txt', 'r') as f:
        sw = [line.strip() for line in f]
    return sw

def load_data(filename,sw):
    with open(filename, 'r', encoding='utf-8') as f:
        texts = [[word for word in line.strip().lower().split() if word not in sw] for line in f]
    return texts

def words_distribution(n_topics,lda,dictionary):
    #主题的词分布
    content=[]
    n_words = 7 # 每个主题显示几个词
    for topic_id in range(n_topics):
        term_distribute_all = lda.get_topic_terms(topicid= topic_id) #terms 不重复的词
        term_distribute = term_distribute_all[ : n_words]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        topic_words=[]
        for t in term_id:
            topic_words.append(dictionary.id2token[t])
        content.append(topic_words)
    return content



def run_main():
    np.set_printoptions(suppress=True)
    t_start = time.time()
    sw= stop_words()
    texts = load_data('news.dat',sw)
    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))
    M = len(texts)
    print('文本数目：%d个' % M)

    print('正在建立词典.')
    dictionary = corpora.Dictionary(texts)
    v = len(dictionary)
    print("词的个数:",v)

    print('正在计算文本向量 --')

    corpus = [dictionary.doc2bow(text) for text in texts]

    print('正在计算文档TF-IDF --')
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立TF-IDF所花费时间{}秒'.format(time.time()-t_start))

    print('LDA模型拟合推断 --')
    n_topics = 10
    t_start = time.time()
    lda = models.LdaModel(corpus= corpus, num_topics= n_topics, id2word=dictionary,
                          alpha = 0.01, eta = 0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)
    print('LDA模型完成,花费时间为{}秒'.format(time.time()- t_start))
    #主题的词分布
    topic_words =words_distribution(n_topics,lda,dictionary)

    # 随机打印某10个文档的主题
    num_show_topic = 10 # 每个文档显示前几个主题
    print('随机打印某5个文档的主题:')
    doc_topics = lda.get_document_topics(corpus_tfidf)# 所有文档的主题分布
    idx = np.arange(M)# M是文档的个数
    np.random.shuffle(idx)
    idx = idx[:5]
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        # topic_idx = topic_distribute.argsort()[:-num_show_topic-1:-1]
        topic_idx = np.argsort(-topic_distribute) #argsort是从小到大
        print(('第%d个文档的前%d个主题(按从大到小排列)：' % (i, num_show_topic)), topic_idx)
        print(topic_distribute[topic_idx])
        print('该文档属于第{}个主题,该文档的词分布是：'.format(topic_idx[0]))
        print(topic_words[topic_idx[0]])


if __name__ == '__main__':
    run_main()
