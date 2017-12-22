import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



if __name__ == '__main__':
	x = np.arange(0.0, 2, 0.01)
	y1 = np.sin(np.pi * x)
	y2 = 1.7 * np.sin(4 * np.pi * x)

	fig = plt.figure()
	axes1 = fig.add_subplot(211)
	axes1.plot(x, y1, x, y2, color='grey')
	axes1.fill_between(x, y1, y2, where=y2 <= y1, facecolors='blue', interpolate=True)
	axes1.fill_between(x, y1, y2, where=y2 >= y1, facecolors='gold', interpolate=True)
	axes1.set_title('Blue where y2<=y1, Gold-color where y2>=y1.')
	axes1.set_ylim(-2, 2)

	y2 = np.ma.masked_greater(y2, 1.0)
	axes2 = fig.add_subplot(212, sharex=axes1)
	axes2.plot(x, y1, x, y2, color='black')
	axes2.fill_between(x, y1, y2, where=y2 <= y1, facecolors='blue', interpolate=True)
	axes2.fill_between(x, y1, y2, where=y2 > y1, facecolors='gold', interpolate=True)
	axes2.set_title('Same as above, but mask')
	axes2.set_ylim(-2, 2)
	axes2.grid('on')

	plt.tight_layout()
	plt.show()
