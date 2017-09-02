# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_data():
    data = datasets.load_iris()
    x,y=data['data'],data['target']
    return x,y

def main():
    x,y = get_data()
    x=x[:,:2]
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)
    model = svm.SVC(C=0.1,kernel='linear',decision_function_shape='ovr')
    model.fit(x_train,y_train)
    y_hat = model.predict(x_test)
    print("训练集准确率是:", accuracy_score(y_train, model.predict(x_train)))
    print("测试集准确率是:",accuracy_score(y_hat,y_test))

    x1_min, x1_max = x[:, 0].min(), x[:, 1].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, 500)
    t2 = np.linspace(x2_min, x2_max, 500)
    x1,x2 = np.meshgrid(t1,t2)
    x_show = np.stack((x1.flat,x2.flat),axis=1)
    # cm_light = mpl.color.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    # cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])




if __name__ == '__main__':
    main()
