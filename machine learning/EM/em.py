# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin



if __name__ == '__main__':
    method = "myself"
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    np.random.seed(0)
    #create data
    mu1_fact = (0,0,0)
    cov1_fact = np.diag((1,2,3))
    data1 = np.random.multivariate_normal(mu1_fact,cov1_fact,400)

    mu2_fact = (2, 2, 1)
    cov2_fact = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
    data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)

    data = np.vstack((data1, data2))
    y = np.array([True]* 400 + [False]* 100)

    if method == "package":
        #package result
        model = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)
        model.fit(data)
        print('第一类占比:{},第二类占比{}'.format(model.weights_[0],model.weights_[1]))
        print('均值:\n', model.means_)
        print('方差:\n', model.covariances_)
        mu1, mu2 = model.means_
        sigma1, sigma2 = model.covariances_

    else:
        #myself
        n_iteration = 100
        n,d = data.shape
        mu1 = np.random.standard_normal(d)
        print(mu1)
        mu2 = np.random.standard_normal(d)
        print(mu2)
        sigma1 = np.identity(d)#The identity array is a square array with ones on the main diagonal.
        sigma2 = np.identity(d)
        pi = 0.5
        for i in range(n_iteration):
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2 )
            tau1 = pi*norm1.pdf(data)
            tau2 = (1-pi)*norm2.pdf(data)
            gamma = tau1/(tau1+tau2)

            #m-step
            mu1 = np.dot(gamma,data)/np.sum(gamma)
            mu2 = np.dot((1-gamma),data)/np.sum(1-gamma)
            sigma1 = np.dot(gamma*(data-mu1).T,data)/np.sum(gamma)
            sigma2 = np.dot((1-gamma)*(data-mu2).T,data)/np.sum(1-gamma)
            pi = np.sum(gamma)/n
            print(i,":\t", mu1, mu2)
        print('类别概率:\t', pi)
        print('均值:\t', mu1, mu2)
        print('方差:\n', sigma1, '\n\n', sigma2, '\n')


    #predict
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')#projection非常重要,不然会报错
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    print(order)
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)
    plt.suptitle(u'EM算法的实现', fontsize=21)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.show()
