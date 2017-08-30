# -*- encoding:utf-8 -*-
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def get_data():
    data = pd.read_csv('wine.data',header=None)
    pd.set_option('display.line_width',1000 )
    print("原始数据:",data.head(5))
    y, x = np.split(data, (1,), axis=1)
    return  x, y

def main():
    x,y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    lr=LogisticRegression(penalty='l2')
    lr.fit(x_train,y_train)
    y_hat_lr=lr.predict(x_test)
    print('Logistic回归正确率：', accuracy_score(y_test, y_hat_lr))


    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    params = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(params, train_data, num_boost_round=2, evals=watch_list)
    y_hat = bst.predict(test_data)
    print('XGBoost正确率：', accuracy_score(y_test, y_hat))


if __name__ == '__main__':
    main()