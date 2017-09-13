# -*- coding:utf-8 -*-

import numpy as np
from numpy.random.mtrand import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


def get_data():
    np.random.seed(0)
    cov1 = np.diag((1,2))
    N1 = 500
    N2 = 300
    N = N1 + N2
    x1 = multivariate_normal(mean=(1,2), cov = cov1, size = N1 )
    m = np.array(((1,2),(1,3)))
    x1 = x1.dot(m)
    x2 = multivariate_normal(mean=(-1, 10), cov=cov1, size = N2)
    x = np.vstack((x1,x2))
    y = np.array([True]*500 + [False]* 300)
    return x ,y

if __name__ == '__main__':
    x, y = get_data()
    types = ('spherical', 'diag', 'tied', 'full')
    err = np.empty(len(types))
    bic = np.empty(len(types))

    for i, type in enumerate(types):
        model = GaussianMixture(n_components=2,random_state=0)
        model.covariance_type = type
        model.fit(x)
        y_hat = model.predict(x)
        accuracy =accuracy_score(y_hat.ravel(),y.ravel())
        if accuracy > 0.5:
            err[i] = 1 - accuracy
        else:
            err [i] = accuracy
        bic[i] = model.bic(x)
    print('错误率:', err.ravel())
    print('BIC:', bic.ravel())
    xpos = np.arange(4)
    plt.figure(facecolor='w')
    ax = plt.axes()
    b1 = ax.bar(xpos-0.3, err, width=0.3, color='#77E0A0')
    b2 = ax.twinx().bar(xpos, bic, width=0.3, color='#FF8080')
    plt.grid(True)
    bic_min, bic_max = expand(bic.min(), bic.max())
    plt.ylim((bic_min, bic_max))
    plt.xticks(xpos, types)
    plt.legend([b1[0], b2[0]], (u'错误率', u'BIC'))
    plt.title(u'不同方差类型的误差率和BIC', fontsize=18)

    optimal = bic.argmin()
    gmm = GaussianMixture(n_components=2,covariance_type= types[optimal], random_state=0)
    gmm.fit(x)
    print("均值:\n", gmm.means_)
    print('协方差矩阵:\n', gmm.covariances_)
    y_hat = gmm.predict(x)

    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    x1_min,x1_max = x[:,0].min(), x[:,0].max()
    x2_min,x2_max = x[:,1].min(), x[:,1].max()
    t1 = np.linspace(x1_min,x1_max,500)
    t2 = np.linspace(x2_min, x2_max, 500)
    x1, x2 = np.meshgrid(t1,t2)
    grid = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = gmm.predict(grid)
    grid_hat = grid_hat.reshape(x1.shape)
    if gmm.means_[0][0] > gmm.means_[1][0]:
        z = grid_hat ==0
        grid_hat[z] == 1
        grid_hat[~z] == 0

    plt.figure(figsize=(9,7))

    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light )
    plt.scatter(x[:,0], x[0:,1], c =y ,marker= 'o', cmap= cm_dark, edgecolors='k', s = 30)

    x1_min,x1_max = expand(x1_min,x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min,x1_max))
    plt.ylim((x2_min,x2_max))
    plt.title("'GMM调参：covariance_type={}".format(types[optimal]), fontsize=20)
    plt.grid()
    plt.show()
