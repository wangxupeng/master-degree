import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def process_signals(x, y):
	return (1 - (x ** 2 + y ** 2)) * np.exp(-y ** 3 / 3)

if __name__ == '__main__':
	t = range(1000)
	y = [np.sqrt(i) for i in t]
	plt.plot(t, y, color='red', lw=2)
	plt.fill_between(t, y, color='silver')
	plt.show()
