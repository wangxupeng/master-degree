#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    data = datasets.load_iris()
    x = np.array(data['data'])
    y = np.array(data['target'])
    feature_names = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    features = [0,1]
    x = x[:,features]
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

    priors = np.array((1,3,5), dtype=float)
    priors /= priors.sum()
    gnb = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=1)),
        ('clf', GaussianNB(priors=priors))])    # 由于鸢尾花数据是样本均衡的，其实不需要设置先验值
    # gnb = KNeighborsClassifier(n_neighbors=3).fit(x, y.ravel())
    gnb.fit(x, y.ravel())
    y_hat = gnb.predict(x)
    print('训练集准确度: %.2f%%' % (100 * accuracy_score(y, y_hat)))
    y_test_hat = gnb.predict(x_test)
    print('测试集准确度：%.2f%%' % (100 * accuracy_score(y_test, y_test_hat)))  # 画图

    N, M = 500, 500     # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_grid = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_grid_hat = gnb.predict(x_grid)                  # 预测值
    y_grid_hat = y_grid_hat.reshape(x1.shape)
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_grid_hat, cmap=cm_light)     # 预测值的显示
    plt.scatter(x[:, features[0]], x[:, features[1]], c=y, edgecolors='k', s=40, cmap=cm_dark)
    plt.scatter(x_test[:, features[0]], x_test[:, features[1]], c=y_test, marker='^', edgecolors='k', s=100, cmap=cm_dark)

    plt.xlabel(feature_names[features[0]], fontsize=13)
    plt.ylabel(feature_names[features[1]], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'GaussianNB对鸢尾花数据的分类结果', fontsize=18)
    plt.grid(True)
    plt.show()
