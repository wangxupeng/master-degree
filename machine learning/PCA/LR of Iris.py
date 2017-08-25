import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


def data_preprocessing():
    data = pd.read_csv("iris.data",header=None) #标准化
    data[4] = pd.Categorical(data[4]).codes #把鸢尾花三个类编写成1,2,3
    x, y = np.split(data.values,(4,), axis=1)
    x = x[:, :2]# 仅使用前两列特征
    return x,y

def main():
    x,y =data_preprocessing()
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=2)),
                   ('clf', LogisticRegression())])
    lr.fit(x, y.ravel())# numpy.ravel Return a contiguous flattened array(返回一个扁平的序列
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    np.set_printoptions(suppress=True) #Small results can be suppressed 小的数字不会用科学计数法输出
    print('y_hat = \n', y_hat)
    print('y_hat_prob = \n', y_hat_prob)
    print(u'准确度：%.2f%%' % (100 * np.mean(y_hat == y.ravel())))

if __name__ == "__main__":
    main()