import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.set_random_seed(4)   ### 3, 
from time import time 


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
		for i in range(self.train_iter):
			t1 = time()
			data, data_len, label = self.TrainData.next()
			loss = self.model.train(data, label, data_len)
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
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			output_prob = self.model.evaluate(data, data_len)
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
			data, data_len, label = self.TrainData.next()
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

	def test(self):
		return LearningBase.test(self)

	def train(self):
		batch_num = self.TrainData.num_of_iter_in_a_epoch
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
	from stream import Create_Multihot_Data_MIMIC3 as data_fn	
	from model_tf import MultihotRnnBase as model_fn
	learn_base = LearningBase(config_fn, data_fn, model_fn)
	learn_base.train()
	'''

	#### Truven; multihot-RNN; next-visit prediction
	from config import get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_Rnn_next_visit as model_fn
	learn_base = LearningBase_truven(config_fn, data_fn, model_fn)
	learn_base.train()


	#### MIMIC; multihot-RETAIN
	'''
	from config import get_multihot_rnn_MIMIC3_config as config_fn
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
	'''
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config as config_fn
	from stream import Create_TF_Multihot_Dictionary_MIMIC as data_fn	
	from model_tf import Multihot_Rnn_Dictionary as model_fn
	learn_base = LearningDictionary(config_fn, data_fn, model_fn)
	learn_base.train()
	'''












