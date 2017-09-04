# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def get_data():
    data = pd.read_csv('bipartition.txt',delimiter='\t')
    data = np.array(data)
    x,y= np.split(data, (2,), axis=1)
    return x,y

def main():
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x,y = get_data()
    models_params = (('linear',0.1), ('linear', 0.5), ('linear',1), ('linear',2),
                     ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                     ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))


    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max()
    t1 = np.linspace(x1_min,x1_max,500)
    t2 = np.linspace(x2_min,x2_max,500)
    x1,x2 = np.meshgrid(t1,t2)
    x_show = np.stack((x1.flat,x2.flat),axis = 1)
    plt.figure(figsize=(14, 10), facecolor='w')
    for i , params in enumerate(models_params):
        models = svm.SVC(C= params[1], kernel= params[0])
        if params == 'rbf':
            models.gamma = params[2]
            title = u'高斯核, C = {}, Gamma = {}'.format(params[1],params[2])
        else:
            title = u'高斯核, C = {}'.format(params[1])

        models.fit(x,y)
        y_hat = models.predict(x)
        accuracy = accuracy_score(y,y_hat)
        print('--------------------------------{}--------------------------------'.format(title))
        print('支撑向量的数目：', models.n_support_)
        print('支撑向量的系数：', models.dual_coef_)
        print('支撑向量：', models.support_)

        plt.subplot(3, 4, i + 1)
        grid_hat = models.predict(x_show)
        grid_hat = grid_hat.reshape(x1.shape)

        plt.pcolormesh(x1, x2, grid_hat,cmap= cm_light, alpha=0.8)
        plt.scatter(x[:,0], x[:,1], c=y, cmap= cm_dark,edgecolors='k', s=40)
        plt.scatter(x[models.support_,0], x[models.support_,1], edgecolors='k', facecolors='none', s=100, marker='o')
        z = models.decision_function(x_show)
        z = z.reshape(x1.shape)
        plt.contour(x1, x2, z, colors=list('kbrbk'), linestyles=['--', '--', '-', '--', '--'],
                    linewidths=[1, 0.5, 1.5, 0.5, 1], levels=[-1, -0.5, 0, 0.5, 1])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title, fontsize=14)
    plt.suptitle(u'SVM不同参数的分类', fontsize=20)
    plt.tight_layout(1.4)
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == '__main__':
    main()
