import numpy as np
import matplotlib.pylab as plt
import warnings


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    x = np.random.randn(1000)
    y1 = np.random.randn(len(x))
    y2 = 1.2 + np.exp(x)

    ax1 = plt.subplot(121)
    plt.scatter(x, y1, color= 'indigo', alpha=0.3,
                edgecolors='white', label='no correl')
    plt.xlabel('no correlation')
    plt.grid(True)
    plt.legend()

    ax2 = plt.subplot(122, sharey=ax1, sharex=ax1) #共享X，Y坐标
    plt.scatter(x, y2, color='green', alpha=0.3,
                edgecolors='grey', label='correl')
    plt.xlabel('strong correlation')
    plt.grid(True)
    plt.legend()

    plt.show()

