# -*- coding:utf-8 -*-

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d

def get_data():
    data = pd.read_csv('HeightWeight.csv')
    data = np.array(data.values)
    y, x = np.split(data,(1,),axis=1)
    return  x, y



if __name__ == '__main__':
    x , y =get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.6, random_state=0)
    gmm = GaussianMixture(n_components=2,covariance_type='full',random_state=0)
    gmm.fit(x_train)
    print("均值:\n", gmm.means_)
    print("协方差矩阵:\n", gmm.covariances_)
    y_train_hat = gmm.predict(x_train)
    y_test_hat = gmm.predict(x_test)
    change = (gmm.means_[0][0] > gmm.means_[1][0])
    if change:
        z = y_train_hat == 0
        y_train_hat[z] = 1
        y_train_hat[~z] = 0
        z = y_test_hat == 0
        y_test_hat[z] = 1
        y_test_hat[~z] = 0
    print('训练集准确率:', accuracy_score(y_train,y_train_hat))
    print('测试集准确率:', accuracy_score(y_test,y_test_hat))

    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    x1_min, x1_max = x_train[:, 0].min(), x_train[:, 0].max()
    x2_min, x2_max = x_train[:, 1].min(), x_train[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    t1 = np.linspace(x1_min,x1_max,500)
    t2 = np.linspace(x2_min,x2_max,500)
    x1,x2 = np.meshgrid(t1,t2)
    grid_test = np.stack((x1.flat,x2.flat),axis=1)
    grid_hat = gmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    if change:
        z = grid_hat ==0
        grid_hat[z] == 1
        grid_hat[~z] == 0
    plt.figure(figsize=(9,7))
    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light)
    plt.scatter(x_train[:,0] , x_train[:,1], c = y_train, s = 40, marker='o', cmap= cm_dark, edgecolors='k')
    plt.scatter(x_test[:,0] , x_test[:,1], c = y_test, s = 40, marker='^',cmap= cm_dark, edgecolors='k')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel(u'身高(cm)', fontsize='large')
    plt.ylabel(u'体重(kg)', fontsize='large')
    plt.title(u'EM算法估算GMM的参数', fontsize=20)
    plt.grid()
    plt.show()