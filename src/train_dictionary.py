from __future__ import print_function
import torch
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np 
from time import time
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)

from config import get_multihot_rnn_config, get_multihot_dictionary_rnn_config, get_pearl_config, get_mnist_dictionary_config
#from stream import TrainFile, TestFile, Create_Multihot_Data, Create_Multihot_Dictionary_Data, Create_Pearl_Data
from stream import Create_Multihot_Data, Create_Multihot_Dictionary_Data, Create_Pearl_Data, MNIST_Data

#from stream import admis_dim as input_dim
#from stream import batch_size as global_batch_size
from model_torch import Multihot_RNN, Multihot_Dictionary_RNN, Multihot_Pearl, faster_Multihot_Dictionary_RNN, MNIST_base, MNIST_dictionary

def test_multihot_rnn(TestData, multihot_rnn):
	y_pred = []
	y_label = []
	for i in range(TestData.num_of_iter_in_a_epoch):
		data, data_len, label = TestData.next()
		output = multihot_rnn(data, data_len)
		#print(output.shape)
		output = torch.softmax(output, dim = 1)
		y_pred.extend([float(output[j][1]) for j in range(output.shape[0])])
		y_label.extend(label)
	return roc_auc_score(y_label, y_pred)



def main_multihot_rnn():
	config = get_multihot_rnn_config()
	### hyperparameter
	train_iter = config['train_iter']
	LR = config['LR']
	multihot_rnn = Multihot_RNN(**config)
	#print(multihot_rnn.rnn_out_dimen)
	#TrainData = Create_Multihot_Data(TrainFile, batch_size = config['batch_size'])
	TrainData = Create_Multihot_Data(is_train = True, **config)
	TestData = Create_Multihot_Data(is_train = False, **config)
	#TestData = Create_Multihot_Data(TestFile, batch_size = config['batch_size'])
	opt_ = torch.optim.SGD(multihot_rnn.parameters(), lr=LR)
	loss_crossentropy = torch.nn.CrossEntropyLoss()
	epoch = 0
	for i in range(train_iter):
		if i % TrainData.num_of_iter_in_a_epoch == 0:
			auc = test_multihot_rnn(TestData, multihot_rnn)
			print('Epoch {}, AUC: {}'.format(epoch, auc), end = '  ')
			if i > 0:
				print('cost {} sec'.format(int(time() - t1)))
			epoch += 1
			t1 = time()	
		data, data_len, label = TrainData.next()
		output = multihot_rnn(data, data_len)
		label = Variable(torch.from_numpy(np.array(label)).long())
		loss = loss_crossentropy(output, label)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]

def test_multihot_dictionary_rnn(TestData, predict_model):
	y_pred = []
	y_label = []
	for i in range(TestData.num_of_iter_in_a_epoch):
		data, _, data_len, label = TestData.next()
		_, output, __ = predict_model(data, data_len)
		#print(output.shape)
		output = torch.softmax(output, dim = 1)
		y_pred.extend([float(output[j][1]) for j in range(output.shape[0])])
		y_label.extend(label)
	return roc_auc_score(y_label, y_pred)


def main_multihot_dictionary_rnn():
	config = get_multihot_dictionary_rnn_config()
	train_iter = config['train_iter']
	LR = config['LR']
	multihot_dictionary_rnn = Multihot_Dictionary_RNN(**config)
	#TrainData = Create_Multihot_Dictionary_Data(TrainFile, batch_size = config['batch_size'], admis_dim = config['input_dim'])
	#TestData = Create_Multihot_Dictionary_Data(TestFile, batch_size = config['batch_size'], admis_dim = config['input_dim'])
	TrainData = Create_Multihot_Dictionary_Data(is_train = True, **config)
	TestData = Create_Multihot_Dictionary_Data(is_train = False, **config)
	opt_ = torch.optim.SGD(multihot_dictionary_rnn.parameters(), lr=LR)
	loss_crossentropy = torch.nn.CrossEntropyLoss()
	epoch = 0 
	for i in range(train_iter):
		if i % TrainData.num_of_iter_in_a_epoch == 0:
			auc = test_multihot_dictionary_rnn(TestData, multihot_dictionary_rnn)
			print('Epoch {}, AUC: {}'.format(epoch, auc), end = ', ')
			if i > 0:
				print('{} sec'.format(int(time() - t1)))
			epoch += 1
			t1 = time()
			if i > 0:
				print('classify loss: {}, reconstruction error: {}'.format(loss_CE.data, loss_recon.data))
			loss_CE, loss_recon = 0, 0

			#print(multihot_dictionary_rnn.check_dictionary)
		data, data_1d_mat, data_len, label = TrainData.next()
		recon, classify_output, code = multihot_dictionary_rnn(data, data_len)
		label = Variable(torch.from_numpy(np.array(label)).long())
		loss1 = loss_crossentropy(classify_output, label) 
		loss2 = config['reconstruction_coefficient'] * F.binary_cross_entropy(recon, data_1d_mat) #####
		loss = loss1 + loss2
		loss_CE += loss1
		loss_recon += loss2

		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]


def test_pearl_rnn(TestData, multihot_rnn):
	y_pred = []
	y_label = []
	for i in range(TestData.num_of_iter_in_a_epoch):
		data, data_len, label = TestData.next()
		output = multihot_rnn(data, data_len)
		#print(output.shape)
		output = torch.softmax(output, dim = 1)
		y_pred.extend([float(output[j][1]) for j in range(output.shape[0])])
		y_label.extend(label)
	return roc_auc_score(y_label, y_pred)


def main_pearl():
	config = get_pearl_config()
	train_iter = config['train_iter']
	LR = config['LR']

	train_num = config['train_num']
	batch_size = config['batch_size']
	assignment = [list(range(400 * i, 400 * (i+1))) for i in range(30)]
	
	pearl = Multihot_Pearl(assignment, **config)
	#TrainData = Create_Multihot_Data(TrainFile, batch_size = batch_size)
	#TestData = Create_Multihot_Data(TestFile, batch_size = batch_size)
	TrainData = Create_Pearl_Data(is_train = True, **config)
	TestData = Create_Pearl_Data(is_train = False, **config)	
	opt_ = torch.optim.SGD(pearl.parameters(), lr=LR)
	loss_crossentropy = torch.nn.CrossEntropyLoss()
	epoch = 0
	print('begin training')
	#t1 = time()
	X_in_all, X_len_all, _ = TrainData.get_all()
	#print('get all data cost {} seconds'.format(int(time() - t1)))
	for i in range(train_iter):
		print('iter {}'.format(i))
		if i % TrainData.num_of_iter_in_a_epoch == 0:
			auc = test_pearl_rnn(TestData, pearl)
			print('Epoch {}, AUC: {}'.format(epoch, auc), end = '  ')
			if i > 0:
				print('cost {} sec'.format(int(time() - t1)))
			epoch += 1
			t1 = time()	
		data, data_len, label = TrainData.next()
		t1 = time()
		pearl.generate_prototype(X_in_all, X_len_all)  #### generate prototype 
		print('generating prototype cost {} seconds'.format(time() - t1))
		output = pearl(data, data_len)  ### update train
		label = Variable(torch.from_numpy(np.array(label)).long())
		loss = loss_crossentropy(output, label)
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		loss_value = loss.data[0]


def evaluate_mnist(TestData, nn):
	TestData.restart()
	iter_num = TestData.batch_num
	Label = []
	Label_pred = []
	for i in range(iter_num):
		feat, lbl = TestData.next()
		lbl_pred = nn(feat, lbl)
		lbl_pred = list(lbl_pred.argmax(1).numpy())
		lbl = list(lbl.numpy())
		Label.extend(lbl)
		Label_pred.extend(lbl_pred)
	accuracy = sum([Label_pred[i] == j for i,j in enumerate(Label)]) * 1.0 / len(Label_pred)
	return accuracy


def evaluate_mnist_dictionary(TestData, nn):
	TestData.restart()
	iter_num = TestData.batch_num
	Label = []
	Label_pred = []
	for i in range(iter_num):
		feat, lbl = TestData.next()
		_, lbl_pred, __ = nn(feat, lbl)
		lbl_pred = list(lbl_pred.argmax(1).numpy())
		lbl = list(lbl.numpy())
		Label.extend(lbl)
		Label_pred.extend(lbl_pred)
	accuracy = sum([Label_pred[i] == j for i,j in enumerate(Label)]) * 1.0 / len(Label_pred)
	return accuracy


def train_mnist_base():
	config = get_mnist_dictionary_config()
	train_iter = config['train_iter']
	LR = config['LR']

	nn = MNIST_base(**config)
	TrainData = MNIST_Data(is_train = True, **config)
	TestData = MNIST_Data(is_train = False, **config)
	opt_ = torch.optim.SGD(nn.parameters(), lr=LR)
	loss_crossentropy = torch.nn.CrossEntropyLoss()	

	for i in range(train_iter):
		#print(loss)
		if i % TrainData.batch_num == 0:
			accuracy = evaluate_mnist(TestData, nn)
			print('accuracy {}'.format(accuracy))
			# pass 		
		feat, lbl = TrainData.next()
		lbl_pred = nn(feat, lbl)
		loss = loss_crossentropy(lbl_pred, lbl)
		opt_.zero_grad()
		loss.backward()
		opt_.step()

def train_mnist_dictionary():
	config = get_mnist_dictionary_config()
	train_iter = config['train_iter']
	LR = config['LR']
	scale = config['scale'] 
	lambda1 = config['loss_lambda1']


	nn = MNIST_dictionary(**config)
	TrainData = MNIST_Data(is_train = True, **config)
	TestData = MNIST_Data(is_train = False, **config)
	opt_ = torch.optim.SGD(nn.parameters(), lr=LR)
	loss_crossentropy = torch.nn.CrossEntropyLoss()	
	loss_mse = torch.nn.MSELoss()
	for i in range(train_iter):
		feat, lbl = TrainData.next()
		recon, classify_output, code = nn(feat, lbl)
		loss1 = loss_crossentropy(classify_output, lbl)
		feat2D = feat.view(feat.shape[0], -1)
		loss2 = loss_mse(recon, feat2D / scale)
		loss = loss1 + lambda1 * loss2
		loss = loss2  
		opt_.zero_grad()
		loss.backward()
		opt_.step()
		if i % TrainData.batch_num == 0:
			accuracy = evaluate_mnist_dictionary(TestData, nn)
			print('accuracy {}'.format(accuracy))
			print('classify loss: {}, reconstruction error: {}'.format(loss1.data, loss2.data))








if __name__ == '__main__':
	#main_multihot_rnn()
	main_multihot_dictionary_rnn()
	#main_pearl()
	#train_mnist_base()
	#train_mnist_dictionary()
	pass












	

