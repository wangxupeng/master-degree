# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint
import pandas as pd



def text_filter(line):
    stop_list = set('for a of the and to in'.split())
    text = line.strip().lower().split()
    word = [x for x in text if x not in stop_list]
    return word



def run_main():
    with open('LDA_test.txt', 'r') as f:
        texts = [text_filter(line) for line in f]
    print("经过过滤后的文本信息:")
    pprint(texts)

    dictionary = corpora.Dictionary(texts)
    print("词典")
    for id,word in zip(dictionary.keys(),dictionary.values()):
        print("{}:{}".format(id,word))
    count_words=[dictionary.doc2bow(text) for text in texts]# 是把文档 doc变成一个稀疏向量，[(0, 1)]，表明id为0的词汇出现了1次
    print('每个文档中的词在该文档中出现的个数(词的编号是所有文档,计数是词在该文档):')
    for line in count_words:
        print(line)
    cw_tfidf = models.TfidfModel(count_words)[count_words]
    print('词的TF-IDF:')
    for word in cw_tfidf:
        print(word)

    print("LSI model:")
    lsi = models.LsiModel(corpus=cw_tfidf, num_topics=2,
                          id2word=dictionary)#Create Dictionary from an existing corpus
    topic_result = [a for a in lsi[cw_tfidf]]
    pprint(topic_result)

    print("LSI的主题:")
    pprint(lsi.print_topics(num_topics=2, num_words=5))
    similarity = similarities.MatrixSimilarity(lsi[cw_tfidf])   # similarities.Similarity()
    print('相似度:')
    pprint(list(similarity))

    print('\nLDA Model:')
    lda = models.LdaModel(corpus=cw_tfidf, num_topics=2, id2word=dictionary, alpha='auto',
                          eta='auto', minimum_probability=0.001,
                          passes=10 )#模型做多少回
    doc_topic = [doc for doc in lda[cw_tfidf]]
    print('主题分布:')
    for doc_topic in lda.get_document_topics(cw_tfidf):
        print(doc_topic)
    for topic_id in range(2):
        print('主题{}的词分布:'.format(topic_id+1))
        pprint(lda.show_topic(topic_id))
    similar = similarities.MatrixSimilarity(lda[cw_tfidf])
    print('文档相似度:')
    for i in similar:
        print(list(i))


if __name__ == '__main__':
    run_main()
