import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(np.c_[iris['data'],iris['target'] ])
    x,y= np.split(data.values,(4,),axis=1)
    x = x[:,:2]
    return x ,y


def first_plot(x,y,model,x_test,y_test):
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    N, M = 50, 50
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)  # Return evenly spaced numbers over a specified interval.
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # Return coordinate matrices from coordinate vectors.
    x_show = np.stack((x1.flat, x2.flat  # flat:A 1-D iterator over the array.
                       ), axis=1)
    print(x_show.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    print(y_show_hat.shape)
    print(y_show_hat)
    y_show_hat = y_show_hat.reshape(x1.shape)
    print(y_show_hat)
    plt.figure()
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 色块
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark,
                marker='*')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

def tree_depth(x_train,y_train,x_test,y_test):
    depth = np.arange(1, 15)
    overfitting=[]
    for d in depth:
        dtc = DecisionTreeClassifier(criterion='entropy',max_depth=d)
        dtc.fit(x_train,y_train)
        y_test_hat=dtc.predict(x_test)
        accuracy = accuracy_score(y_test_hat, y_test)
        of = 1-accuracy
        overfitting.append(of)
        print("如果树的深度是{}".format(d),"它的准确率是:%.2f%%"%(100*accuracy))
    plt.figure()
    plt.plot(depth,overfitting,'ro-',linewidth=2)
    plt.xlabel(u'树的深度',fontsize=15)
    plt.ylabel(u'过拟合程度',fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()

def main():
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x,y = get_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=1)
    print(y_test.shape)
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train,y_train)
    y_test_hat = model.predict(x_test)
    accuracy = accuracy_score(y_test_hat,y_test)
    print("准确率是:",accuracy)

    first_plot(x, y, model, x_test, y_test)#决策树分类图

    tree_depth(x_train, y_train, x_test, y_test)

if __name__ =="__main__":
    main()
