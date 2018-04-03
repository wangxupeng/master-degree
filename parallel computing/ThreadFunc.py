# -*- coding: utf-8 -*-
from time import sleep, ctime
import threading

loops = [4, 2]


class ThreadFunc(object):
	def __init__(self, func, args):
		self.func = func
		self.args = args

	def __call__(self):
		self.func(*self.args)


def loop(nloop, nsec):
	print('start loop', nloop, 'at:', ctime())
	sleep(nsec)
	print('loop', nloop, 'done at:', ctime())

def main():
	print('starting at:', ctime())
	threads = []
	nloops = range(len(loops))

	for i in nloops:  # create all threads
		t = threading.Thread(
			target=ThreadFunc(loop, (i, loops[i])))
		threads.append(t)

	for i in nloops:
		threads[i].start()

	for i in nloops:
		threads[i].join()  # 程序挂起，直到线程结束

	print('all done at:', ctime())


if __name__ == '__main__':
	main()
