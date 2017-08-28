import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

def get_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(np.c_[iris['data'],iris['target'] ])
    x,y= np.split(data.values,(4,),axis=1)
    return x ,y

def main():
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    x_prime,y = get_data()
    feature_pairs = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
    plt.figure(figsize=(10, 9))

    for i, pair in enumerate(feature_pairs):
        x=x_prime[:,pair]
        clf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=3)
        clf.fit(x,y.ravel())
        N, M = 50, 50
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
        t1 = np.linspace(x1_min, x1_max, N)  # Return evenly spaced numbers over a specified interval.
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)  # Return coordinate matrices from coordinate vectors.
        x_show = np.stack((x1.flat, x2.flat  # flat:A 1-D iterator over the array.
                           ), axis=1)
        y_hat = clf.predict(x)
        accuracy = accuracy_score(y,y_hat)
        print("特征为{},{}".format(iris_feature[pair[0]],iris_feature[pair[1]]), "它的准确率是:%.2f%%" % (100 * accuracy))

        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_hat = clf.predict(x_show)  # 预测值
        y_hat = y_hat.reshape(x1.shape)
        plt.subplot(2, 3, i + 1)
        plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
        plt.scatter(x[:,0],x[:,1],c=y,edgecolors="k",cmap=cm_dark)
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min,x1_max)
        plt.ylim(x2_min,x2_max)
        plt.grid(True)
    plt.tight_layout(2.5)
    plt.subplots_adjust(top=0.92)
    plt.suptitle(u'随机森林对鸢尾花数据的两特征组合的分类结果', fontsize=18)
    plt.show()


if __name__ =="__main__":
    main()
