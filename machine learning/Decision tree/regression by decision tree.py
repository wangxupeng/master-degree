import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeRegressor

def get_data():
    N = 100
    x = np.random.rand(N) * 6 - 3     # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的
    return x,y

def plot1(x,y,x_test,y_hat):
    plt.plot(x, y, 'r*', ms=10, label='Actual')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def main():
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x,y = get_data()
    dt =DecisionTreeRegressor(criterion='mse',max_depth=9)
    dt.fit(x,y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plot1(x, y, x_test, y_hat)

    depth = [2,4,6,8,10]
    clr = 'rgbmy'
    plt.plot(x, y, 'ko', ms=6, label='Actual')
    for d,c in zip(depth,clr):
        dtr =DecisionTreeRegressor(criterion='mse',max_depth=d)
        dtr.fit(x,y)
        y_predict = dtr.predict(x_test)
        plt.plot(x_test, y_predict, '-', color = c, linewidth=2, label="树的深度{}".format(d))
    plt.legend(loc="upper right")
    plt.title(u'树的深度与预测效果')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()