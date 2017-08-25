# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint

def data_plot(data,x,y):
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # 绘制1
    plt.figure(facecolor='w')#底色为白色
    plt.plot(data['TV'], y, 'ro', label='TV')#红色圆点
    plt.plot(data['Radio'], y, 'g^', label='Radio')#绿色上三角
    plt.plot(data['Newspaper'], y, 'bv', label='Newspaer')#蓝色下三角
    plt.legend(loc='lower right')
    plt.xlabel(u'广告花费', fontsize=16)
    plt.ylabel(u'销售额', fontsize=16)
    plt.title(u'广告花费与销售额对比数据', fontsize=20)
    plt.grid()
    plt.show()

    # 绘制2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

def result_plot(x_test,y_test,y_hat):
    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True)
    plt.show()

def main():
    path = 'Advertising.csv'
    data = pd.read_csv(path)  # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)
    data_plot(data,x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print(type(x_test))
    print(x_train.shape, y_train.shape)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_, linreg.intercept_)
    order = y_test.argsort(axis=0) #因为分类的时候是随机分类的,所以这里为了作图要排序
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('R2 = ', linreg.score(x_train, y_train))
    print('R2 = ', linreg.score(x_test, y_test))
    result_plot(x_test,y_test,y_hat)

if __name__ == "__main__":
    main()


