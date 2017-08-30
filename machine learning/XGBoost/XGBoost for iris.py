# -*- encoding:utf-8 -*-
from sklearn import datasets
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split   # cross_validation


def get_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(np.c_[iris['data'],iris['target'] ])
    x,y= np.split(data.values,(4,),axis=1)
    return x ,y


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def main():
    x,y = get_data()
    x_train,x_test,y_train,y_text = train_test_split(x,y,random_state=1,test_size=50)
    data_train = xgb.DMatrix(x_train,label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_text)
    watch_list = [(data_test,'eval'),(data_train,'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param,data_train,num_boost_round=6,evals=watch_list)
    y_hat = bst.predict(data_test)
    accuracy = accuracy_score(y_hat,y_text)
    print("准确率:",accuracy)




if __name__ =="__main__":
    main()
