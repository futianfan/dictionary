import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.set_random_seed(3)



def test(model, All_data):
	from sklearn.metrics import roc_auc_score
	batch_num = All_data.num_of_iter_in_a_epoch
	label_all = []
	predict_all = [] 
	for i in range(batch_num):
		next_data = All_data.next()
		data, data_len, label = next_data[0], next_data[1], next_data[2]
		#data, data_len, label = All_data.next()
		output_prob = model.evaluate(data, data_len)
		output_prob = output_prob[0]
		output_prob = [i[1] for i in output_prob]
		#print(output_prob)
		label_all.extend(label)
		predict_all.extend(output_prob)

	return roc_auc_score(label_all, predict_all)


def train_multihot_rnn():
	from model_tf import MultihotRnnBase
	from stream import Create_Multihot_Data
	from config import get_multihot_rnn_TF_config

	config = get_multihot_rnn_TF_config()	
	train_iter = config['train_iter']
	TrainData = Create_Multihot_Data(is_train = True, **config)
	TestData = Create_Multihot_Data(is_train = False, **config)

	multihot_rnn_base = MultihotRnnBase(**config)


	batch_num = TrainData.num_of_iter_in_a_epoch
	total_loss = 0
	for i in range(train_iter):
		## data 
		data, data_len, label = TrainData.next()
		## train & train loss
		loss = multihot_rnn_base.train(data, label, data_len)
		total_loss += loss 
		if i > 0 and i % batch_num == 0:
			auc = test(multihot_rnn_base, TestData)
			print('Loss: {}, test AUC {}.'.format(total_loss, auc))
			total_loss = 0


def train_multihot_rnn_MIMIC():
	from model_tf import MultihotRnnBase
	from stream import Create_Multihot_Data_MIMIC3
	from config import get_multihot_rnn_MIMIC3_config

	config = get_multihot_rnn_MIMIC3_config()	
	train_iter = config['train_iter']
	TrainData = Create_Multihot_Data_MIMIC3(is_train = True, **config)
	TestData = Create_Multihot_Data_MIMIC3(is_train = False, **config)

	multihot_rnn_base = MultihotRnnBase(**config)


	batch_num = TrainData.num_of_iter_in_a_epoch
	total_loss = 0
	for i in range(train_iter):
		## data 
		data, data_len, label = TrainData.next()
		## train & train loss
		loss = multihot_rnn_base.train(data, label, data_len)
		total_loss += loss 
		if i > 0 and i % batch_num == 0:
			auc = test(multihot_rnn_base, TestData)
			print('Loss: {}, test AUC {}.'.format(total_loss / batch_num, auc))
			total_loss = 0


def train_multihot_Attention_rnn_MIMIC():
	from model_tf import Multihot_Rnn_Attention
	from stream import Create_Multihot_Data_MIMIC3
	from config import get_multihot_rnn_MIMIC3_config

	config = get_multihot_rnn_MIMIC3_config()	
	train_iter = config['train_iter']
	TrainData = Create_Multihot_Data_MIMIC3(is_train = True, **config)
	TestData = Create_Multihot_Data_MIMIC3(is_train = False, **config)

	multihot_rnn_base = Multihot_Rnn_Attention(**config)


	batch_num = TrainData.num_of_iter_in_a_epoch
	total_loss = 0
	for i in range(train_iter):
		## data 
		data, data_len, label = TrainData.next()
		## train & train loss
		loss = multihot_rnn_base.train(data, label, data_len)
		total_loss += loss 
		if i > 0 and i % batch_num == 0:
			auc = test(multihot_rnn_base, TestData)
			print('Loss: {}, test AUC {}.'.format(total_loss / batch_num, auc))
			total_loss = 0



def train_multihot_rnn_dictionary_HeartFailure():
	"""
		multihot + dictionary + heart-failure  
	"""
	from model_tf import Multihot_Rnn_Dictionary
	from stream import Create_TF_Multihot_Dictionary_Data ### Create_Multihot_Data
	from config import get_multihot_rnn_dictionary_TF_config

	config = get_multihot_rnn_dictionary_TF_config()
	train_iter = config['train_iter']
	TrainData = Create_TF_Multihot_Dictionary_Data(is_train = True, **config)
	TestData = Create_TF_Multihot_Dictionary_Data(is_train = False, **config)

	multihot_rnn_dictionary = Multihot_Rnn_Dictionary(**config)
	batch_num = TrainData.num_of_iter_in_a_epoch
	total_classify_loss, total_recon_loss, total_dictionary_loss = 0, 0, 0
	for i in range(train_iter):
		data, data_len, label, data_recon = TrainData.next()
		classify_loss, recon_loss, dictionary_loss = multihot_rnn_dictionary.train(data, label, data_len, data_recon)
		total_classify_loss += classify_loss
		total_recon_loss += recon_loss
		total_dictionary_loss += dictionary_loss
		if i > 0 and i % batch_num == 0:
			total_classify_loss /= batch_num
			total_recon_loss /= batch_num
			total_dictionary_loss /= batch_num
			auc = test(multihot_rnn_dictionary, TestData)
			print('classify Loss:{}, recon loss:{}, dictionary loss:{}, test AUC {}.'.format(
				str(total_classify_loss)[:6], str(total_recon_loss)[:7], str(total_dictionary_loss)[:7], str(auc)[:6]))
			total_classify_loss, total_recon_loss, total_dictionary_loss = 0.0, 0.0, 0.0

	output = multihot_rnn_dictionary.generation_prototype_patient()
	np.save(config['prototype_npy'], output)

def train_multihot_rnn_dictionary_MIMIC():
	"""
		feature + model + data 
		multihot + dictionary + MIMIC 
	"""
	from model_tf import Multihot_Rnn_Dictionary
	from stream import Create_TF_Multihot_Dictionary_MIMIC ## Create_TF_Multihot_Dictionary_Data ### Create_Multihot_Data
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config  ## get_multihot_rnn_dictionary_TF_config

	config = get_multihot_rnn_dictionary_TF_MIMIC3_config()
	train_iter = config['train_iter']
	TrainData = Create_TF_Multihot_Dictionary_MIMIC(is_train = True, **config)
	TestData = Create_TF_Multihot_Dictionary_MIMIC(is_train = False, **config)

	multihot_rnn_dictionary = Multihot_Rnn_Dictionary(**config)
	batch_num = TrainData.num_of_iter_in_a_epoch
	total_classify_loss, total_recon_loss, total_dictionary_loss = 0, 0, 0
	for i in range(train_iter):
		data, data_len, label, data_recon = TrainData.next()
		classify_loss, recon_loss, dictionary_loss = multihot_rnn_dictionary.train(data, label, data_len, data_recon)
		total_classify_loss += classify_loss
		total_recon_loss += recon_loss
		total_dictionary_loss += dictionary_loss
		if i > 0 and i % batch_num == 0:
			total_classify_loss /= batch_num
			total_recon_loss /= batch_num
			total_dictionary_loss /= batch_num
			auc = test(multihot_rnn_dictionary, TestData)
			print('classify Loss:{}, recon loss:{}, dictionary loss:{}, test AUC {}.'.format(
				str(total_classify_loss)[:6], str(total_recon_loss)[:7], str(total_dictionary_loss)[:7], str(auc)[:6]))
			total_classify_loss, total_recon_loss, total_dictionary_loss = 0.0, 0.0, 0.0
	output = multihot_rnn_dictionary.generation_prototype_patient()
	np.save(config['prototype_npy'], output)





if __name__ == '__main__':
	#train_multihot_rnn()
	train_multihot_rnn_dictionary_HeartFailure()
	#train_multihot_rnn_MIMIC()
	#train_multihot_Attention_rnn_MIMIC()
	#train_multihot_rnn_dictionary_MIMIC()
	





