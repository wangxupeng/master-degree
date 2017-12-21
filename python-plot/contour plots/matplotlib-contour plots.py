import numpy as np
import matplotlib.pylab as plt
import warnings
import matplotlib as mpl

def process_signal(x, y):
    return (1 - (x ** 2 + y ** 2)) * np.exp(-y ** 3 / 3)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    x = np.arange(-1.5, 1.5, 0.1)
    y = np.arange(-1.5, 1.5, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = process_signal(X, Y)

    N = np.arange(-1, 1.5, 0.3)

    cs = plt.contour(Z, N, linewidths=2, cmap=mpl.cm.jet)
    plt.clabel(cs, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(cs)

    plt.title('My function: $z=(1-x^2+y^2) e^{-(y^3)/3}$')
    plt.show()
