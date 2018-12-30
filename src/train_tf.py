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
		data, data_len, label = All_data.next()
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





if __name__ == '__main__':
	train_multihot_rnn()





