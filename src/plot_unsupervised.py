import matplotlib.pyplot as plt
import numpy as np


def plot_inner(x, y, fig_name):

	z = list(range(len(x)))	
	color_list = 'bgrcmy'
	for idx, acc in enumerate(y):
		leng = len(acc)
		zz = z[-leng:]
		plt.plot(zz, acc, color_list[idx] + '-.', label = str(x[idx]) + '% labelled data')

	_ = plt.xticks(z, x)

	plt.xlabel('Fraction of Total data (%)', fontsize = 16)
	plt.ylabel('Average Accuracy (AUC)', fontsize = 16)
	plt.legend()
	plt.savefig(fig_name)


def plot_HF():
	x = [1, 5, 10, 50, 100]
	y = [[0.5377, 0.5429, 0.5712, 0.5943, 0.6005], \
		 [0.6166, 0.6189, 0.6231, 0.6247], \
		 [0.6304, 0.6322, 0.6345], \
		 [0.6546, 0.6565]
	]
	plot_inner(x, y, './result/HF_1.png')

'''
def plot_HF():
	x = [1, 5, 10, 50, 100]
	y = [[0.5377, 0.5429, 0.5712, 0.5943, 0.6005], \
		 [0.6166, 0.6189, 0.6231, 0.6247], \
		 [0.6304, 0.6322, 0.6345], \
		 [0.6546, 0.6565]
	]
	z = list(range(len(x)))

	color_list = 'bgrcmy'
	for idx, acc in enumerate(y):
		leng = len(acc)
		zz = z[-leng:]
		plt.plot(zz, acc, color_list[idx] + '-.')

	_ = plt.xticks(z, x)

	plt.xlabel('Fraction of Total data (%)', fontsize = 16)
	plt.ylabel('Average Accuracy (AUC)', fontsize = 16)
	plt.legend()

	plt.savefig('./result/HF_1.png')
'''


def plot_mimic():
	x = [2, 5, 10, 50, 100]
	y = [[0.5377, 0.5429, 0.5712, 0.5943, 0.6005], \
		 [0.6166, 0.6189, 0.6231, 0.6247], \
		 [0.6304, 0.6322, 0.6345], \
		 [0.6546, 0.6565]
	]
	plot_inner(x, y, './result/mimic_1.png')	


if __name__ == "__main__":

	plot_HF()

	plot_mimic()



