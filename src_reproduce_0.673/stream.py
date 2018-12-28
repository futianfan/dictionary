import numpy as np 
import os 
from torch.autograd import Variable
import torch 
import struct, gzip 
from config import get_multihot_rnn_config

DataFolder = './data'
TrainFile = 'training_data_1.txt'
TestFile = 'test_data_1.txt'

TrainFile = os.path.join(DataFolder, TrainFile)
TestFile = os.path.join(DataFolder, TestFile)


config = get_multihot_rnn_config()

max_length = config['max_length']
admis_dim = config['input_dim']
batch_size = config['batch_size']


def code_to_visit_level(admissions, timestamp):
	uniq_time = list(set(timestamp))
	uniq_time.sort()
	visit_dict = {time: [] for time in uniq_time}
	for idx, adm in enumerate(admissions):
		time = timestamp[idx]
		visit_dict[time].append(adm)
	return [visit_dict[time] for time in uniq_time][-max_length:]

def line_to_visit_level(line):
	admissions = line.split('\t')[2]
	admissions = [int(i) for i in admissions.split()]
	timestamp = line.split('\t')[3]
	timestamp = [int(i) for i in timestamp.split()]
	assert len(timestamp) == len(admissions)
	return code_to_visit_level(admissions, timestamp)

def lst_to_data(lst_lst):
	lst_lst = lst_lst[-max_length:]
	leng = len(lst_lst)
	datamat = np.zeros((max_length, admis_dim))
	for i in range(leng):
		for j in lst_lst[i]:
			datamat[i,j] = 1
	return datamat, leng

def batch_lst_to_data(batch_lst_lst):
	batch_datamat = list(map(lst_to_data, batch_lst_lst))
	batch_leng = [i[1] for i in batch_datamat]
	batch_datamat = [i[0] for i in batch_datamat]
	batch_size = len(batch_datamat)
	x,y = batch_datamat[0].shape
	for i in range(batch_size):
		datamat = np.concatenate([datamat, batch_datamat[i].reshape(1,x,y)], 0) if i > 0 else batch_datamat[i].reshape(1,x,y)
	return datamat, batch_leng

def label_1d_to_2d(label):
	leng = len(label)
	class_num = 2
	label_2d = np.zeros((leng, class_num))
	for i in range(leng):
		label_2d[i, label[i]] = 1
	return label_2d


class Create_Multihot_Data(object):
	"""
	create batch data
	"""
	def __init__(self, is_train = True, **config):
		#self.max_length = max_length
		filename = config['train_file'] if is_train else config['test_file']
		batch_size = config['batch_size']
		with open(filename, 'r') as fin:
			lines = fin.readlines()[1:]
			self.label = list(map(lambda x: 1 if x.split('\t')[0] == 'True' else 0, lines ))
			self.data_lst = list(map(line_to_visit_level, lines))
			del lines
		self.batch_size = batch_size
		self.total_num = len(self.label)
		self.batch_num = int(np.ceil(self.total_num / self.batch_size))
		self.batch_id = 0 
		self.random_shuffle = np.arange(self.total_num)  ### no shuffle at first epoch 

	@property
	def check_label(self):
		return self.label[:10]

	@property
	def total_N(self):
		return self.total_num

	@property
	def num_of_iter_in_a_epoch(self):
		return self.batch_num 

	###	 TO DO:	 __next__
	def next0(self):
		bgn = self.batch_id * self.batch_size
		endn = bgn + self.batch_size
		self.batch_id += 1
		if self.batch_id > self.batch_num - 1:
			np.random.shuffle(self.random_shuffle)
			self.batch_id = 0
		indx = self.random_shuffle[bgn:endn]
		return [self.data_lst[i] for i in indx], [self.label[i] for i in indx]
		#data, label = self.data_lst[bgn:endn], self.label[bgn:endn]
		#return data, label

	def next(self):
		data, label = self.next0()
		data, batch_leng = batch_lst_to_data(data)
		return data, batch_leng, label ### data:numpy array;  label is list
'''
def lst_to_multihot_vec(vec_size, lst):
	vec = [1 if i in lst else 0 for i in range(vec_size)] 
	return np.array(vec)
'''
class Create_Multihot_Dictionary_Data(Create_Multihot_Data):
	#def __init__(self, fin, batch_size, admis_dim):
	def __init__(self, is_train = True, **config):
		Create_Multihot_Data.__init__(self, is_train, **config)
		from functools import reduce
		f = lambda line: reduce(lambda x,y:x+y, line)
		self.data_lst_1d = list(map(f, self.data_lst))
		self.admis_dim = config['input_dim']

	def lst_to_multihot_vec(self, lst):
		vec = [1 if i in lst else 0 for i in range(self.admis_dim)] 
		return np.array(vec)


	@property
	def first_5_line(self):
		return self.data_lst_1d[:5]

	def next(self):
		data, label = Create_Multihot_Data.next0(self)	### same with Create_Multihot_Data

		from functools import reduce
		f = lambda line: reduce(lambda x,y:x+y, line)
		data_lst1d = list(map(f, data))
		data_lst1d_mat = list(map(self.lst_to_multihot_vec, data_lst1d))
		data_lst1d_mat = [i.reshape(1,-1) for i in data_lst1d_mat]
		data_lst1d_mat = np.concatenate(data_lst1d_mat, axis = 0) 
		data_lst1d_mat = Variable(torch.FloatTensor(data_lst1d_mat))

		data, batch_leng = batch_lst_to_data(data)
		return data, data_lst1d_mat, batch_leng, label 

class Create_Pearl_Data(Create_Multihot_Data):
	def __init__(self, is_train = True, **config):
		Create_Multihot_Data.__init__(self, is_train, **config)
		self.input_dim = config['input_dim']
		self.max_length = config['max_length']
		self.weight = [1.0 for i in range(self.total_num)]

	def next0(self):
		bgn = self.batch_id * self.batch_size
		endn = bgn + self.batch_size
		self.batch_id += 1
		if self.batch_id > self.batch_num - 1:
			self.batch_id = 0
		indx = self.random_shuffle[bgn:endn]
		return [self.data_lst[i] for i in indx], [self.label[i] for i in indx]

	def get_all(self):
		tmp = self.batch_id
		self.batch_id = 0
		seq_leng = []
		label = []
		data = np.zeros((self.total_num, self.max_length, self.input_dim), dtype = np.float)
		for i in range(self.batch_num):
			if i % 10 == 0: print(i)
			batch_data, batch_leng, batch_label = self.next()
			#data = np.concatenate([data, batch_data], 0) if i > 0 else batch_data
			data[i * self.batch_size: i * self.batch_size + self.batch_size] = batch_data
			seq_leng.extend(batch_leng) 
			label.extend(batch_label) 
		self.batch_id = tmp 
		return data, seq_leng, label
		### np.arr, list, list 

	def update_weight(self):
		pass 

class MNIST_Data:
	def __init__(self, is_train = True, **config):
		self.is_train = is_train
		self.feature_file = config['train_feature'] if is_train else config['test_feature']
		self.label_file = config['train_label'] if is_train else config['test_label']
		self.batch_size = config['batch_size']
		self.class_num = config['num_class']

		with open(self.label_file, 'rb') as flbl:
			magic, num = struct.unpack(">II", flbl.read(8))
			self.lbl = np.fromfile(flbl, dtype=np.int8)

		with open(self.feature_file, 'rb') as fimg:
			magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
			self.img = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.lbl), rows, cols)

		### numpy => torch
		self.img = Variable(torch.FloatTensor(self.img))
		self.lbl = Variable(torch.LongTensor(self.lbl.astype(dtype = np.uint8)))

		self.total_num, self.rows, self.cols = self.img.shape
		self.batch_id = 0
		self.batch_num = int(np.ceil(self.total_num / self.batch_size))

	def restart(self):
		self.batch_id = 0

	def next(self):
		bgn, endn = self.batch_id * self.batch_size, (1+self.batch_id)*self.batch_size
		self.batch_id += 1
		if self.batch_id == self.batch_num:	self.batch_id = 0
		return self.img[bgn:endn], self.lbl[bgn:endn]






if __name__ == '__main__':
	from config import get_multihot_rnn_config, get_multihot_dictionary_rnn_config, get_pearl_config, get_mnist_dictionary_config
	'''
	### 1. Create_Multihot_Data
	TrainData = Create_Multihot_Data(TrainFile, batch_size = batch_size)
	TestData = Create_Multihot_Data(TestFile, batch_size = batch_size)
	#print(TrainData.check_label)
	for i in range(10000):
		print(i)
		data, data_len, label = TrainData.next()	
	'''
	'''
	### 2. Create_Multihot_Dictionary_Data
	TrainData = Create_Multihot_Dictionary_Data(TrainFile, batch_size = batch_size, admis_dim = admis_dim)
	print(TrainData.first_5_line)
	_, d, _, _ = TrainData.next()
	print(d.sum(1))
	'''

	### 3. Create_Pearl_Data
	'''
	config = get_pearl_config()
	TrainData = Create_Pearl_Data(is_train = True, **config)
	from time import time
	t1 = time()
	'''

	### 4. MNIST    
	config = get_mnist_dictionary_config()
	TrainData = MNIST_Data(is_train = True, **config)
	TestData = MNIST_Data(is_train = False, **config)
	for i in range(100):
		a, __ = TrainData.next()
		print(a.max())









