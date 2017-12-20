import numpy as np
import matplotlib.pylab as plt
import warnings


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    plt.figure(1, figsize=(6,6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])
    labels = 'Spring', 'Summer', 'Autumn', 'Winter'

    x = [15, 30, 45, 10]
    explode = (0.1, 0.1, 0.1, 0.1)
    plt.pie(x,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%', startangle=67)
    plt.title('Rainy days by season')

    plt.show()
