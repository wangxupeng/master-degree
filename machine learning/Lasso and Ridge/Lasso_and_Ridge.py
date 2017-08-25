# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

def read_data():
    data = pd.read_csv('Advertising.csv')
    x = data[['TV','Radio','Newspaper']]
    y = data ['Sales']
    return x,y

def order(x_test,y_test):
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    return x_test , y_test

def predict_plot(x_test,y_test,y_hat):
    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor="w")
    plt.plot(t,y_test,"k-",linewidth=2,label=u'实际值')
    plt.plot(t,y_hat,"c-.",linewidth=2,label=u'预测值')
    plt.title('预测值和实际值的比较',fontsize=15)
    plt.legend(loc="upper right")
    plt.show()

def main():
    x,y = read_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.75)#强行从1开始,所以每次随机都是相同的
    #model = Lasso()
    model = Ridge()
    alpha_can = np.logspace(-3,2,10)
    np.set_printoptions(suppress=True)
    lasso_model = GridSearchCV(model,param_grid={'alpha':alpha_can},cv=10) #cross validation alpha就是正则化里的λ
    lasso_model.fit(x_train,y_train)
    print("最好的λ:",lasso_model.best_params_)
    ################################################################################################
    x_test,y_test=order(x_test,y_test)
    y_hat = lasso_model.predict(x_test)
    print("R-squre:",lasso_model.score(x_test,y_test))#R-squre
    MSE = np.average((y_hat-np.array(y_test))**2)
    RMSE = np.sqrt(MSE)
    print("MSE:",MSE)
    print("RMSE",RMSE)
    predict_plot(x_test,y_test,y_hat)




if __name__ == "__main__":
    main()