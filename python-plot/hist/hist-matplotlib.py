import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    mu = 100
    sigma = 15
    x = np.random.normal(mu, sigma, 10000)

    # the histogram of the data
    plt.hist(x, bins=35, color='r',edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Histogram:}\ \mu=%d,\ \sigma=%d$' % (mu,
                                                                 sigma))
    plt.show()
