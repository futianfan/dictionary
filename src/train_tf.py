import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.set_random_seed(4)   ### 3, 
from time import time 


def fidelity_compute(lst1, lst2):
	assert len(lst1) == len(lst2)
	from sklearn.metrics import roc_auc_score
	lst3 = lst1[:]
	lst3.sort()
	fraction = 0.5 ## first 50% 
	idx = int(len(lst3) * fraction)
	threshold = lst3[idx]
	lst4 = list(map(lambda x:1 if x > threshold else 0, lst1))
	return roc_auc_score(lst4, lst2)	



class LearningBase:
	"""
		can be used for 
			(1) HeartFailure; Multihot-Rnn
			(2) MIMIC; multihot-Rnn
			(3) MIMIC; multihot-RETAIN

	"""


	def __init__(self, config_fn, data_fn, model_fn):
		self.config = config_fn()
		self.TrainData = data_fn(is_train = True, **self.config)
		self.TestData = data_fn(is_train = False, **self.config)
		self.model = model_fn(**self.config)
		self.train_iter = self.config['train_iter']



	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		total_loss, epoch, total_time = 0.0, 1, 0.0 
		best_auc = 0 
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label = self.TrainData.next()
			loss = self.model.train(data, label, data_len)
			total_time += time() - t1
			total_loss += loss 
			if i > 0 and i % batch_num == 0:
				auc = self.test()
				if auc > best_auc:
					best_auc = auc 
				print('Epoch {}, Loss: {}, test AUC {}, best AUC {}, time: {} sec'.format(
					epoch, 
					str(total_loss / batch_num)[:5], 
					str(auc)[:5], 
					str(best_auc)[:5],
					str(total_time)[:5]))
				epoch += 1
				total_time, total_loss = 0.0, 0.0



	def test(self):
		from sklearn.metrics import roc_auc_score
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			output_prob = self.model.evaluate(data, data_len)
			output_prob = output_prob[0]
			output_prob = [i[1] for i in output_prob]
			label_all.extend(label)
			predict_all.extend(output_prob)
		return roc_auc_score(label_all, predict_all)

class LearningAggregate(LearningBase):
	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		total_loss, epoch, total_time = 0.0, 1, 0.0 
		for i in range(self.train_iter):
			t1 = time()
			data, label = self.TrainData.next()
			loss = self.model.train(data, label)
			total_time += time() - t1
			total_loss += loss 
			if i > 0 and i % batch_num == 0:
				auc = self.test()
				print('Epoch {}, Loss: {}, test AUC {}, time: {} sec'.format(
					epoch, 
					str(total_loss / batch_num)[:5], 
					str(auc)[:5], 
					str(total_time)[:5]))
				epoch += 1
				total_time, total_loss = 0.0, 0.0

	def test(self):
		from sklearn.metrics import roc_auc_score
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, label = next_data[0], next_data[1]
			output_prob = self.model.evaluate(data)
			output_prob = output_prob[0]
			output_prob = [i[1] for i in output_prob]
			label_all.extend(label)
			predict_all.extend(output_prob)
		return roc_auc_score(label_all, predict_all)



class LearningBase_truven(LearningBase):
	def __init__(self, config_fn, data_fn, model_fn):
		LearningBase.__init__(self, config_fn, data_fn, model_fn)
		self.topk = self.config['topk']

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		total_loss, epoch, total_time = 0.0, 1, 0.0 
		for i in range(self.train_iter):
			t1 = time()
			next_data = self.TrainData.next()
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			loss = self.model.train(data, label, data_len)
			total_time += time() - t1
			total_loss += loss 

			if i >= 0 and i % batch_num == 0:
				recall = self.test()
				print('Epoch {}, Loss: {}, test-recall: {} time: {} sec'.format(
					epoch, 
					str(total_loss / batch_num)[:5], 
					str(recall)[:6],
					str(total_time)[:5]))
				epoch += 1
				total_time, total_loss = 0.0, 0.0

	def test(self):
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		total_label_number = 0
		total_correct_number = 0
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			output_prob = self.model.evaluate(data, data_len)
			output_prob = output_prob[0]
			prediction = [list(i[-self.topk:]) for i in np.argsort(output_prob,1)]
			bs = len(label)
			for j in range(bs):
				true_label = set(label[j])
				predict_label = set(prediction[j])
				total_label_number += len(true_label)
				total_correct_number += len(true_label.intersection(predict_label))
		return total_correct_number / total_label_number



class LearningDictionary(LearningBase):
	"""
		used for 
			(1) HeartFailure; multihot-dictionary
			(2) MIMIC; multihot-dictionary
	"""
	def __init__(self, config_fn, data_fn, model_fn):
		LearningBase.__init__(self, config_fn, data_fn, model_fn)
		self.topk = self.config['topk']

	'''	
	def test(self):
		auc_score =  LearningBase.test(self)

		batch_num = self.TestData.num_of_iter_in_a_epoch
		total_correct_number, total_label_number = 0, 0 
		for i in range(batch_num):
			next_data = self.TestData.next_1()
			data, data_len, label, data_recon = next_data[0], next_data[1], next_data[2], next_data[3]
			output_prob = self.model.evaluate(data, data_len)
			X_recon = output_prob[1]
			prediction = [list(i[-self.topk:]) for i in np.argsort(X_recon,1)]
			bs = len(label)
			for j in range(bs):
				total_correct_number += len(list(filter(lambda x:x in data_recon[j], prediction)))
				total_label_number += len(data_recon[j])
			

		return auc_score, total_correct_number * 1.0 / total_label_number		
	'''

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		epoch, total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 1, 0, 0, 0, 0
		best_auc = 0 
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label, data_recon = self.TrainData.next()
			classify_loss, recon_loss, dictionary_loss = self.model.train(data, label, data_len, data_recon)
			total_time += time() - t1 
			total_classify_loss += classify_loss
			total_recon_loss += recon_loss
			total_dictionary_loss += dictionary_loss
			if i > 0 and i % batch_num == 0:
				total_classify_loss /= batch_num
				total_recon_loss /= batch_num
				total_dictionary_loss /= batch_num
				#auc, recall = self.test()
				auc = self.test()
				#recon_loss_lst.append(total_recon_loss)
				print('Epoch {}, classify Loss:{}, recon loss:{}, dictionary obj loss:{}, test AUC {}, time: {} sec'.format(
					epoch, 
					str(total_classify_loss)[:6], 
					str(total_recon_loss)[:7], 
					str(total_dictionary_loss)[:7], 
					str(auc)[:6],
					str(total_time)[:4])
					)
				epoch += 1
				total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 0.0, 0.0, 0.0, 0.0
		## save prototype patient 
		#output = self.model.generation_prototype_patient()
		#np.save(self.config['prototype_npy'], output)
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		for j in range(batch_num):
			data, data_len, _, _ = self.TrainData.next()
			sparse_code = self.model.generate_sparse_code(data, data_len)
			sparse_code_all = np.concatenate((sparse_code_all, sparse_code), 0) if j > 0 else sparse_code
		sparse_code_all = sparse_code_all.transpose()
		print(sparse_code_all.shape)
		u, _, _ = np.linalg.svd(sparse_code_all, full_matrices = False)
		u = u.transpose()
		output = self.model.generation_prototype_patient(u)
		np.save(self.config['prototype_npy'], output)



class SemiSupervised_LearningDictionary(LearningDictionary):
	'''
		MIMIC
	'''
	def __init__(self, config_fn, data_fn, model_fn):
		LearningBase.__init__(self, config_fn, data_fn, model_fn)
		self._data_split()

		'''
		self.config = config_fn()
		self.TrainData = data_fn(is_train = True, **self.config)
		self.TestData = data_fn(is_train = False, **self.config)
		self.model = model_fn(**self.config)
		self.train_iter = self.config['train_iter']
		'''
	def _data_split(self):
		### split data  supervised / unsupervised
		## config -> supervised_train
		supervised_ratio = self.config['supervised_ratio']
		total_ratio = self.config['total_ratio']
		train_file = self.config['train_file']
		supervised_train = self.config['supervised_train']
		unsupervised_train = self.config['unsupervised_train']
		lines = open(train_file, 'r').readlines()
		random_shuffle = np.arange(len(lines))
		np.random.shuffle(random_shuffle)
		lines = [lines[i] for i in random_shuffle]
		with open(supervised_train, 'w') as fout:
			for line in lines[:int(len(lines) * supervised_ratio)]:
				fout.write(line)
		with open(unsupervised_train, 'w') as fout:
			for line in lines[int(len(lines) * supervised_ratio): int(len(lines) * total_ratio)]:
				fout.write(line)

		self.config['train_file'] = self.config['supervised_train']
		self.SupervisedTrainData = data_fn(is_train = True, **self.config)
		self.config['train_file'] = self.config['unsupervised_train']
		self.UnsupervisedTrainData = data_fn(is_train = True, **self.config)


	def train(self):
		batch_num = self.SupervisedTrainData.num_of_iter_in_a_epoch
		epoch, total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 1, 0, 0, 0, 0
		best_auc = 0.0 
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label, data_recon = self.SupervisedTrainData.next()
			classify_loss, recon_loss, dictionary_loss = self.model.train(data, label, data_len, data_recon)
			data, data_len, _, data_recon = self.UnsupervisedTrainData.next()
			recon_loss2, dictionary_loss2 = self.model.UnsupervisedTrain(data, data_len, data_recon)
			total_time += time() - t1 
			total_classify_loss += classify_loss
			total_recon_loss += recon_loss
			total_dictionary_loss += dictionary_loss
			if i > 0 and i % batch_num == 0:
				total_classify_loss /= batch_num
				total_recon_loss /= batch_num
				total_dictionary_loss /= batch_num
				#auc, recall = self.test()
				auc = self.test()
				if auc > best_auc:
					best_auc = auc
				#recon_loss_lst.append(total_recon_loss)
				print('Epoch {}, classify Loss:{}, recon loss:{}, dictionary obj loss:{}, test AUC {}, best AUC {}, time: {} sec'.format(
					epoch, 
					str(total_classify_loss)[:6], 
					str(total_recon_loss)[:7], 
					str(total_dictionary_loss)[:7], 
					str(auc)[:6],
					str(best_auc)[:6], 
					str(total_time)[:4])
					)
				epoch += 1
				total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 0.0, 0.0, 0.0, 0.0

class SemiSupervised_LearningDictionary_HF(SemiSupervised_LearningDictionary):
	'''
		Heart Failure:  data format processing
	'''
	def __init__(self, config_fn, data_fn, model_fn):
		LearningBase.__init__(self, config_fn, data_fn, model_fn)


		supervised_ratio = self.config['supervised_ratio']
		total_ratio = self.config['total_ratio']
		train_file = self.config['train_file']
		supervised_train = self.config['supervised_train']
		unsupervised_train = self.config['unsupervised_train']
		lines = open(train_file, 'r').readlines()
		lines = lines[1:]
		random_shuffle = np.arange(len(lines))
		np.random.shuffle(random_shuffle)
		lines = [lines[i] for i in random_shuffle]
		with open(supervised_train, 'w') as fout:
			fout.write('Heart Failure\n')
			for line in lines[:int(len(lines) * supervised_ratio)]:
				fout.write(line)
		with open(unsupervised_train, 'w') as fout:
			fout.write('Heart Failure\n')
			for line in lines[int(len(lines) * supervised_ratio): int(len(lines) * total_ratio)]:
				fout.write(line)

		self.config['train_file'] = self.config['supervised_train']
		self.SupervisedTrainData = data_fn(is_train = True, **self.config)

		self.config['train_file'] = self.config['unsupervised_train']
		self.UnsupervisedTrainData = data_fn(is_train = True, **self.config)



class LearningDictionary2(LearningDictionary):
	"""
		used for fidelity ???
		used for 
			(1) HeartFailure; multihot-dictionary
			(2) MIMIC; multihot-dictionary
	"""

	def __init__(self, config_fn, data_fn, model_fn, fidelity_model):
		self.config = config_fn()
		self.TrainData = data_fn(is_train = True, **self.config)
		self.TestData = data_fn(is_train = False, **self.config)
		self.model = model_fn(**self.config)
		self.train_iter = self.config['train_iter']
		self.fidelity_model = fidelity_model(**self.config)

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		epoch, total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 1, 0, 0, 0, 0
		fidelityloss_all = 0
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label, data_recon = self.TrainData.next()
			classify_loss, recon_loss, dictionary_loss, output_recon = self.model.train(data, label, data_len, data_recon)
			fidelityloss = self.fidelity_model.train(output_recon, label)
			fidelityloss_all += fidelityloss
			total_time += time() - t1 
			total_classify_loss += classify_loss
			total_recon_loss += recon_loss
			total_dictionary_loss += dictionary_loss
			if i > 0 and i % batch_num == 0:
				total_classify_loss /= batch_num
				total_recon_loss /= batch_num
				total_dictionary_loss /= batch_num
				#auc, recall = self.test()
				auc, fidelity_accu = self.test2()
				#recon_loss_lst.append(total_recon_loss)
				print('Epoch {}, classify Loss:{}, recon loss:{}, dictionary obj loss:{}, test AUC {}, fidelity auc {}, fidelity loss {}, time: {} sec'.format(
					epoch, 
					str(total_classify_loss)[:6], 
					str(total_recon_loss)[:7], 
					str(total_dictionary_loss)[:7], 
					str(auc)[:6],
					str(fidelity_accu)[:6],
					str(fidelityloss_all)[:6],
					str(total_time)[:4])
					)
				epoch += 1
				total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 0.0, 0.0, 0.0, 0.0
				fidelityloss_all = 0
		## save prototype patient 
		#_, output_recon_all = self.test2()
		#print(output_recon_all.shape)

	def test2(self):
		from sklearn.metrics import roc_auc_score
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		fidelity_all = []
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			output = self.model.evaluate2(data, data_len)
			output_prob, output_recon = output[0], output[1]	
			output_prob = [i[1] for i in output_prob]
			output_prob_fidelity = self.fidelity_model.evaluate(output_recon)[0]
			output_prob_fidelity = [i[1] for i in output_prob_fidelity]
			label_all.extend(label)
			predict_all.extend(output_prob)
			fidelity_all.extend(output_prob_fidelity)
		return roc_auc_score(label_all, predict_all),  fidelity_compute(predict_all, fidelity_all)




class LearningDictionaryFidelity(LearningDictionary):
	"""
		fidelity
		used for 
			(1) HeartFailure; multihot-dictionary
			(2) MIMIC; multihot-dictionary
	"""
	def __init__(self, config_fn, data_fn, model_fn, fidelity_model):
		LearningDictionary.__init__(self, config_fn, data_fn, model_fn)
		self.fidelity_model = fidelity_model(**self.config)


	'''	
	def test(self):
		auc_score =  LearningBase.test(self)

		batch_num = self.TestData.num_of_iter_in_a_epoch
		total_correct_number, total_label_number = 0, 0 
		for i in range(batch_num):
			next_data = self.TestData.next_1()
			data, data_len, label, data_recon = next_data[0], next_data[1], next_data[2], next_data[3]
			output_prob = self.model.evaluate(data, data_len)
			X_recon = output_prob[1]
			prediction = [list(i[-self.topk:]) for i in np.argsort(X_recon,1)]
			bs = len(label)
			for j in range(bs):
				total_correct_number += len(list(filter(lambda x:x in data_recon[j], prediction)))
				total_label_number += len(data_recon[j])
			

		return auc_score, total_correct_number * 1.0 / total_label_number		
	'''

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		epoch, total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 1, 0, 0, 0, 0
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label, data_recon = self.TrainData.next()
			classify_loss, recon_loss, dictionary_loss = self.model.train(data, label, data_len, data_recon)
			output_recon = self.model.evaluate2(data, data_len)

			total_time += time() - t1 
			total_classify_loss += classify_loss
			total_recon_loss += recon_loss
			total_dictionary_loss += dictionary_loss
			if i > 0 and i % batch_num == 0:
				total_classify_loss /= batch_num
				total_recon_loss /= batch_num
				total_dictionary_loss /= batch_num
				#auc, recall = self.test()
				auc = self.test()
				#recon_loss_lst.append(total_recon_loss)
				print('Epoch {}, classify Loss:{}, recon loss:{}, dictionary obj loss:{}, test AUC {}, time: {} sec'.format(
					epoch, 
					str(total_classify_loss)[:6], 
					str(total_recon_loss)[:7], 
					str(total_dictionary_loss)[:7], 
					str(auc)[:6],
					str(total_time)[:4])
					)
				epoch += 1
				total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 0.0, 0.0, 0.0, 0.0
		## save prototype patient 
		output = self.model.generation_prototype_patient()
		np.save(self.config['prototype_npy'], output)



class LearningDictionary_Truven(LearningBase_truven, LearningDictionary):

	def __init__(self, config_fn, data_fn, model_fn):
		LearningBase_truven.__init__(self, config_fn, data_fn, model_fn)

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		recon_loss_lst = []
		epoch, total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 1, 0, 0, 0, 0
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label, data_recon = self.TrainData.next()
			classify_loss, recon_loss, dictionary_loss = self.model.train(data, label, data_len, data_recon)
			total_time += time() - t1 
			total_classify_loss += classify_loss
			total_recon_loss += recon_loss
			total_dictionary_loss += dictionary_loss
			if i > 0 and i % batch_num == 0:
				total_classify_loss /= batch_num
				total_recon_loss /= batch_num
				total_dictionary_loss /= batch_num
				auc = self.test()
				recon_loss_lst.append(total_recon_loss)
				print('Epoch {}, classify Loss:{}, recon loss:{}, dictionary obj loss:{}, test recall {}, time: {} sec'.format(
					epoch, 
					str(total_classify_loss)[:6], 
					str(total_recon_loss)[:7], 
					str(total_dictionary_loss)[:7], 
					str(auc)[:6],
					str(total_time)[:4])
					)
				epoch += 1
				total_classify_loss, total_recon_loss, total_dictionary_loss, total_time = 0.0, 0.0, 0.0, 0.0

			## save prototype patient  basis vector [1, 0 0 0 0],  [0 1 0 0 0 ]  [0 0 1 0 0 ]
			#output = self.model.generation_prototype_patient()
			#np.save(self.config['prototype_npy'], output)
		batch_num = self.TrainData.num_of_iter_in_a_epoch
		for j in range(batch_num):
			data, data_len, _, _ = self.TrainData.next()
			sparse_code = self.model.generate_sparse_code(data, data_len)
			sparse_code_all = np.concatenate((sparse_code_all, sparse_code), 0) if j > 0 else sparse_code
		sparse_code_all = sparse_code_all.transpose()
		print(sparse_code_all.shape)
		u, _, _ = np.linalg.svd(sparse_code_all, full_matrices = False)
		u = u.transpose()
		output = self.model.generation_prototype_patient(u)
		np.save(self.config['prototype_npy'], output)

	def test(self):
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		total_label_number = 0
		total_correct_number = 0
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, data_len, label, data_recon = next_data[0], next_data[1], next_data[2], next_data[3]
			output_prob = self.model.evaluate(data, data_len)
			output_prob = output_prob[0]
			prediction = [list(i[-self.topk:]) for i in np.argsort(output_prob,1)]
			bs = len(label)
			for j in range(bs):
				true_label = set(label[j])
				predict_label = set(prediction[j])
				total_label_number += len(true_label)
				total_correct_number += len(true_label.intersection(predict_label))
		return total_correct_number / total_label_number


class SemiSupervised_LearningDictionary_Truven(LearningDictionary_Truven, SemiSupervised_LearningDictionary):

	def __init__(self, config_fn, data_fn, model_fn):
		LearningDictionary_Truven.__init__(self, config_fn, data_fn, model_fn) 
		self._data_split()

	def train(self):
		SemiSupervised_LearningDictionary.train(self)


if __name__ == "__main__":

	####  HeartFailure; Multihot-Rnn
	'''
	from config import get_multihot_rnn_TF_config as config_fn
	from stream import Create_Multihot_Data as data_fn	
	from model_tf import MultihotRnnBase as model_fn
	learn_base = LearningBase(config_fn, data_fn, model_fn)
	learn_base.train()
	'''


	#### MIMIC; multihot-Rnn
	'''
	from config import get_multihot_rnn_MIMIC3_config as config_fn
	#from config import get_multihot_rnn_MIMIC3_ccs_config as config_fn
	from stream import Create_Multihot_Data_MIMIC3 as data_fn	
	from model_tf import MultihotRnnBase as model_fn
	learn_base = LearningBase(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	#### Truven; multihot-RNN; next-visit prediction
	'''
	from config import get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_Rnn_next_visit as model_fn
	learn_base = LearningBase_truven(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	#### Truven; RETAIN attention; next-visit prediction;  
	'''
	from config import get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_Rnn_Attention_next_visit as model_fn
	learn_base = LearningBase_truven(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	#### MIMIC; multihot-RETAIN
	'''
	#from config import get_multihot_rnn_MIMIC3_config as config_fn
	from config import get_multihot_rnn_MIMIC3_ccs_config as config_fn
	from stream import Create_Multihot_Data_MIMIC3 as data_fn	
	from model_tf import Multihot_Rnn_Attention as model_fn
	learn_base = LearningBase(config_fn, data_fn, model_fn)
	learn_base.train()
	'''



	#### HeartFailure; multihot-dictionary
	'''
	from config import get_multihot_rnn_dictionary_TF_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_Data as data_fn	
	from model_tf import Multihot_Rnn_Dictionary as model_fn
	learn_base = LearningDictionary(config_fn, data_fn, model_fn)
	learn_base.train()
	'''
	

	#### MIMIC; multihot-dictionary
	
	#from config import get_multihot_rnn_dictionary_TF_MIMIC3_config as config_fn
	'''
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_MIMIC as data_fn	
	from model_tf import Multihot_Rnn_Dictionary as model_fn
	learn_base = LearningDictionary(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	### MIMIC, semi-supervised 
	'''
	from config import unsupervised_get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_MIMIC as data_fn 
	from model_tf import SemiSupervised_Multihot_Rnn_Dictionary as model_fn 
	learn_base = SemiSupervised_LearningDictionary(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	### Heart Failure, semi-supervised 
	
	from config import unsupervised_get_multihot_rnn_dictionary_TF_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_Data as data_fn 
	from model_tf import SemiSupervised_Multihot_Rnn_Dictionary as model_fn 
	learn_base = SemiSupervised_LearningDictionary_HF(config_fn, data_fn, model_fn)
	learn_base.train()
	
	
	### Truven; multihot-RNN; next-visit prediction; dictionary 
	'''
	from config import get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_dictionary_next_visit as model_fn
	learn_base = LearningDictionary_Truven(config_fn, data_fn, model_fn)
	learn_base.train()
	''' 
	### Semisupervised  Truven; multihot-RNN; next-visit prediction; dictionary 
	'''
	from config import semisupervised_get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn 
	from model_tf import Semisupervised_Multihot_dictionary_next_visit as model_fn 
	learn_base = SemiSupervised_LearningDictionary_Truven(config_fn, data_fn, model_fn)
	learn_base.train() 
	'''

	### Truven focus on reconstruction 
	'''
	from config import get_dictionary_TF_truven_config_reconstruction as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_dictionary_next_visit as model_fn
	learn_base = LearningDictionary_Truven(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	### aggregate feature; heart failure
	'''
	from config import get_aggregate_config as config_fn
	from stream import Create_Aggregate_heart_failure as data_fn	
	from model_tf import AggregateBase as model_fn
	learn_base = LearningAggregate(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	'''
	from config import get_multihot_rnn_dictionary_TF_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_Data as data_fn	
	from model_tf import Multihot_Rnn_Dictionary as model_fn
	from model_tf import fidelity_network
	learn_base = LearningDictionaryFidelity(config_fn, data_fn, model_fn, fidelity_network)
	learn_base.train()
	'''



	#### compute fidelity MIMIC 3
	'''
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_MIMIC as data_fn	
	from model_tf import Multihot_Rnn_Dictionary2 as model_fn
	from model_tf import fidelity_network as fidelity_model
	learn_base = LearningDictionary2(config_fn, data_fn, model_fn, fidelity_model)
	learn_base.train()
	'''

	#### compute fidelity heart failure
	'''
	from config import get_multihot_rnn_dictionary_TF_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_Data as data_fn	
	from model_tf import Multihot_Rnn_Dictionary2 as model_fn
	from model_tf import fidelity_network as fidelity_model
	learn_base = LearningDictionary2(config_fn, data_fn, model_fn, fidelity_model)
	learn_base.train()
	'''




