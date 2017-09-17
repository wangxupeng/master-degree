# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_data():
    print('开始加载数据.')
    t_start = time()
    remove = ()
    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'
    data_train = fetch_20newsgroups(subset='train',categories=categories,random_state=0,shuffle=True,remove= remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories, random_state=0, shuffle=True, remove=remove)
    t_end = time()
    print('加载时间:',t_start - t_end)
    print('数据类型',type(data_train))
    print('训练集包含文本个数',len(data_train.data))
    print('测试集包含文本个数', len(data_test.data))
    print('类别名称:\n',categories)
    y_train = data_train.target
    y_test = data_test.target
    categories = data_train.target_names
    print('--------展示前十个样本--------')
    for i in range(10):
        print('文本{}属于类别{}'.format(i+1,categories[y_train[i]]))
        print(data_train.data[i])
        print('\n\n\n')
    vectorizer = TfidfVectorizer(input='content',stop_words='english',max_df=0.5,sublinear_tf=True)
    x_train = vectorizer.fit_transform(data_train.data)
    x_test = vectorizer.transform(data_test.data)
    print(u'训练集样本个数：%d，特征个数：%d' % x_train.shape)
    print(u'停止词:\n',)
    pprint(vectorizer.get_stop_words())
    feature_names = np.asarray(vectorizer.get_feature_names())
    return  x_train,y_train,x_test,y_test

def calculate(clf,x_train,y_train,x_test,y_test):
    print('分类器是:',clf)
    alpha_can = np.logspace(-3, 2, 10)
    model = GridSearchCV(clf, param_grid={'alpha':alpha_can}, cv=5)
    m = alpha_can.size
    #  MultinomialNB ,BernoulliNB and RidgeClassifier
    if hasattr(clf, 'alpha'):#判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        model.param_grid = {'alpha':alpha_can}
        m = alpha_can.size
    #  KNeighborsClassifier
    if hasattr(clf, 'n_neighbors'):
        neighbors_can = np.arange(1, 15)
        model.param_grid = {'n_neighbors':neighbors_can}
        m = neighbors_can.size
    #  RandomForestClassifier
    if hasattr(clf, 'max_depth'):
        max_depth_can = np.logspace(4,10)
        model.set_params(param_grid={'max_depth': max_depth_can})
        m = max_depth_can.size
    # SVM
    if hasattr(clf, 'C'):
        C_can = np.logspace(1, 3, 3)
        gamma_can = np.logspace(-3, 0, 3)
        model.set_params(param_grid={'C':C_can, 'gamma':gamma_can})
        m = C_can.size * gamma_can.size

    t_start = time()
    model.fit(x_train,y_train)
    t_end = time()
    t_train = (t_end - t_start) / (5*m)
    print('5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), m, t_train))
    print('最优超参数为：', model.best_params_)
    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print(u'测试时间：%.3f秒' % t_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print(u'测试集准确率：%.2f%%' % (100 * acc))
    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]     # 去掉末尾的Classifier
    if name == 'SVC':
        name = 'SVM'
    return t_train, t_test, 1-acc, name




def main():
    x_train, y_train,x_test, y_test= get_data()
    print(u'\n\n===================\n分类器的比较：\n')
    clfs = (MultinomialNB(),
              BernoulliNB(),
              KNeighborsClassifier(),
              RidgeClassifier(),
              RandomForestClassifier(),
              SVC())
    result = []
    for clf in clfs:
        test_result = calculate(clf,x_train,y_train,x_test,y_test)
        result.append(test_result)
        print('\n\n')
    result = np.array(result)
    time_train, time_test, err, names = result.T
    time_train = time_train.astype(np.float)
    time_test = time_test.astype(np.float)
    err = err.astype(np.float)

    # plot
    index = np.arange(len(time_train))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(9,10))
    ax = plt.axes()
    width = 0.25
    b1 = ax.bar(index, err, width = width, color='g')
    ax_t = ax.twinx()
    b2 = ax_t.bar(index+ width, time_train, width = width, color = 'b' )
    b3 = ax_t.bar(index + 2* width, time_test, width = width , color = 'r')
    plt.xticks(index + 2* width, names)
    plt.legend([b1[0], b2[0], b3[0]], (u'错误率', u'训练时间', u'测试时间'), loc='upper left', shadow=True)
    plt.title(u'新闻组文本数据不同分类器间的比较', fontsize=18)
    plt.xlabel(u'分类器名称')
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()




if __name__ == '__main__':
    main()
