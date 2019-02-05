import tensorflow as tf
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.set_random_seed(4)   ### 3, 
from time import time, sleep

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



def load_ckpt(saver, sess, config, ckpt_dir="train"):
	"""Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, 
        waiting 10 secs in the case of failure. Also returns checkpoint name."""
	while True:
		try:
			latest_filename = "checkpoint_best" if ckpt_dir=="valid" else None
			ckpt_dir = os.path.join(config['log_folder'], ckpt_dir)
			ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
			tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
			saver.restore(sess, ckpt_state.model_checkpoint_path)
			print('succss to load')
			return ckpt_state.model_checkpoint_path
		except:
			print('fail to load')
			sleep(10)



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
		saver = tf.train.Saver(max_to_keep = 1)
		if not os.path.exists(self.config['log_train_folder']): os.makedirs(self.config['log_train_folder'])
		supervisor = tf.train.Supervisor(logdir = self.config['log_train_folder'],
										 is_chief = True,
										 summary_op = None,
										 save_summaries_secs = 60,
										 save_model_secs = 5,
										 )
		sess_context_manager = supervisor.prepare_or_wait_for_session()
		with sess_context_manager as sess:
			batch_num = self.TrainData.num_of_iter_in_a_epoch
			total_loss, epoch, total_time = 0.0, 1, 0.0 
			for i in range(self.train_iter):
				t1 = time()
				data, data_len, label = self.TrainData.next()
				loss = self.model.train(data, label, data_len, sess)
				total_time += time() - t1
				total_loss += loss 
				if i > 0 and i % batch_num == 0:
					auc = self.test(sess)
					print('Epoch {}, Loss: {}, test AUC {}, time: {} sec'.format(
						epoch, 
						str(total_loss / batch_num)[:5], 
						str(auc)[:5], 
						str(total_time)[:5]))
					epoch += 1
					total_time, total_loss = 0.0, 0.0


	def test(self, sess):
		from sklearn.metrics import roc_auc_score
		batch_num = self.TestData.num_of_iter_in_a_epoch
		label_all = []
		predict_all = [] 
		for i in range(batch_num):
			next_data = self.TestData.next()
			data, data_len, label = next_data[0], next_data[1], next_data[2]
			output_prob = self.model.evaluate(data, data_len, sess)
			output_prob = output_prob[0]
			output_prob = [i[1] for i in output_prob]
			label_all.extend(label)
			predict_all.extend(output_prob)
		return roc_auc_score(label_all, predict_all)


	def valid(self):
		saver = tf.train.Saver(max_to_keep=1)
		sess = tf.Session()
		if not os.path.exists(self.config['log_valid_folder']): os.makedirs(self.config['log_valid_folder'])
		best_accuracy = None

		while True: 
			_ = load_ckpt(saver, sess, self.config, ckpt_dir = 'train')
			accuracy = self.test(sess)
			
			if best_accuracy == None or accuracy > best_accuracy:
				best_accuracy = accuracy
				saver.save(sess, self.config['bestmodel_save_path'], latest_filename='checkpoint_best')
				print('save checkpoint')
			else:
				print('not update checkpoint', end = '\t')
				print(str(accuracy)[:5], str(best_accuracy)[:8])


	def final_test(self):
		'''
			best-model from valid directory
		'''
		saver = tf.train.Saver(max_to_keep=1)
		sess = tf.Session()
		if not os.path.exists(self.config['log_valid_folder']): raise Exception('valid directory not exists')
		
		_ = load_ckpt(saver, sess, self.config, ckpt_dir = 'valid')
		accuracy = self.test(sess)
		print('test AUC is {}'.format(accuracy))



if __name__ == "__main__":

	####  HeartFailure; Multihot-Rnn	
	from config import get_multihot_rnn_TF_config as config_fn
	from stream import Create_Multihot_Data as data_fn	
	from model_tf import MultihotRnnBase as model_fn
	import sys
	learn_base = LearningBase(config_fn, data_fn, model_fn)
	if sys.argv[1] == 'train':
		learn_base.train()
	elif sys.argv[1] == 'valid':
		learn_base.valid()
	elif sys.argv[1] == 'test':
		learn_base.final_test()
	else:
		raise Exception('sys.argv: train or valid')

	'''
		python src_ckpt/train_tf.py train
		python src_ckpt/train_tf.py valid

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

	### Truven; multihot-RNN; next-visit prediction; dictionary 
	'''
	from config import get_multihot_rnn_dictionary_TF_truven_config as config_fn
	from stream import Create_truven as data_fn	
	from model_tf import Multihot_dictionary_next_visit as model_fn
	learn_base = LearningDictionary_Truven(config_fn, data_fn, model_fn)
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




