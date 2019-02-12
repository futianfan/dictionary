import os

def get_multihot_rnn_config():
	'''
		Heart Failure 

	'''
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(2e6)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

	return config 


def get_aggregate_config():
	'''
		Heart Failure 
		aggregate feature 
	'''
	config = {}
	config['batch_size'] = 8
	config['rnn_hidden_num'] = 300
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(2e6)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

	return config 



def get_multihot_dictionary_rnn_config():
	'''
		Heart Failure 

	'''	
	config = {}
	config['batch_size'] = 8  ### 64, 8 ?  
	config['max_length'] = 5  ### 5,
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['train_iter'] = int(1e6)

	config['fc1'] = 50 
	config['dictionary_size'] = 10 
	config['fc2'] = 200 ### in decoder 
	config['lambda1'] = 1e-2  ### L1 regularization coefficient 
	config['lambda2'] = 1e-1
	config['reconstruction_coefficient'] = 1e-1  ### 1e-2


	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

	return config 

def get_pearl_config():
	config = {}
	config['proc_batch_size'] = 256
	config['batch_size'] = 64

	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2

	config['train_num'] = 13436 
	config['test_num'] = 3358 
	config['LR'] = 1e-1
	config['train_iter'] = int(2e6)


	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

	return config 


def get_mnist_dictionary_config():
	config = {}

	config['data_folder'] = './MNIST_data'
	config['train_feature'] = os.path.join(config['data_folder'], 'train-images-idx3-ubyte')
	config['train_label'] = os.path.join(config['data_folder'], 'train-labels-idx1-ubyte')
	config['test_feature'] = os.path.join(config['data_folder'], 't10k-images-idx3-ubyte')
	config['test_label'] = os.path.join(config['data_folder'], 't10k-labels-idx1-ubyte')

	## train
	config['num_class'] = 10
	config['LR'] = 1e-1
	config['train_iter'] = int(1e6)	
	config['batch_size'] = 512  ### 64, 8 ?  

	### NN parameter
	config['rows'] = 28
	config['cols'] = 28
	config['dim1'] = 100
	config['dim2'] = 50 

	config['scale'] = 255 
	config['loss_lambda1'] = 1e-1


	config['batch_first'] = True
	config['dictionary_size'] = 10 

	config['fc1'] = 50 
	config['fc2'] = 200 ### in decoder 
	config['lambda1'] = 1e-2  ### L1 regularization coefficient 
	config['lambda2'] = 1e-1
	config['reconstruction_coefficient'] = 1e-4  ### 1e-2


	return config

'''
def get_multihot_rnn_MIMIC3_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 942
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(1e4)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'mimic_train')
	config['test_file'] = os.path.join(config['data_folder'], 'mimic_test')

	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'

	return config 
'''



def get_multihot_rnn_MIMIC3_icdcode_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 942
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(1e4)
	config['attention_size'] = 50 

	config['data_folder'] = './data'

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'mimic_train')
	config['test_file'] = os.path.join(config['data_folder'], 'mimic_test')
	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'

	return config 


def get_multihot_rnn_MIMIC3_ccs_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	#config['input_dim'] = 942
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(1e4)
	config['attention_size'] = 50 

	config['data_folder'] = './data'
	config['mapfile'] = os.path.join(config['data_folder'], 'mimic_ccs_idx2text')
	config['input_dim'] = len(open(config['mapfile'], 'r').readlines())
	assert config['input_dim'] == 283

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'MimicCcsTrain')
	config['test_file'] = os.path.join(config['data_folder'], 'MimicCcsTest')
	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'

	return config 


def get_multihot_rnn_MIMIC3_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 942
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(1e4)
	config['attention_size'] = 50 

	config['data_folder'] = './data'

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'mimic_train')
	config['test_file'] = os.path.join(config['data_folder'], 'mimic_test')
	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'

	return config 



def get_multihot_rnn_TF_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-2
	config['test_num'] = 3358 
	config['train_iter'] = int(2e6)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

	return config 

def get_multihot_rnn_dictionary_TF_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 1867
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-2
	config['test_num'] = 3358 
	config['train_iter'] = int(3e4)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')  ### heart-failure
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')
	config['mapfile'] = os.path.join(config['data_folder'], 'heartfailure_code_map.txt')

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'heartfailure_prototype.npy')
	config['prototype_text'] = os.path.join(config['result_folder'], 'heartfailure_prototype_topk.txt')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10
	config['topk'] = 50 

	return config 



def get_multihot_rnn_dictionary_TF_MIMIC3_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['input_dim'] = 942
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = 3000 ##int(9e4)   ### 3e4  
	### batch_size=8 => 750 iter <=> 1 epoch 

	config['data_folder'] = './data'

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'mimic_prototype.npy')
	config['prototype_text'] = os.path.join(config['result_folder'], 'mimic_prototype_topk')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'mimic_train')
	config['test_file'] = os.path.join(config['data_folder'], 'mimic_test')
	config['mapfile'] = os.path.join(config['data_folder'], 'mimic_code_map.txt')

	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'
	config['topk'] = 50 
	return config 




def get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 5  ### 5 future work: try larger max-length  
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['num_class'] = 2
	config['LR'] = 1e-2
	config['test_num'] = 3358 
	config['train_iter'] = int(5e4) ## 7e4 ##int(9e4)   ### 3e4  
	### batch_size=8 => 750 iter <=> 1 epoch 

	config['data_folder'] = './data'

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'mimic_prototype.npy')
	config['prototype_text'] = os.path.join(config['result_folder'], 'mimic_prototype_topk')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10


	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'
	config['topk'] = 50 

	config['mapfile'] = os.path.join(config['data_folder'], 'mimic_ccs_idx2text')
	config['input_dim'] = len(open(config['mapfile'], 'r').readlines())
	assert config['input_dim'] == 283

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'MimicCcsTrain')
	config['test_file'] = os.path.join(config['data_folder'], 'MimicCcsTest')

	return config 


def unsupervised_get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config():
	'''
		MIMIC-3
	'''
	config = get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config()
	config['supervised_ratio'] = 0.2
	config['total_ratio'] = 1.0
	config['supervised_train'] = 'supervised_train'
	config['unsupervised_train'] = 'unsupervised_train'
	assert config['total_ratio'] > config['supervised_ratio']

	return config 


def unsupervised_get_multihot_rnn_dictionary_TF_config():
	'''
		Heart Failure 
	'''
	config = get_multihot_rnn_dictionary_TF_config()
	config['supervised_ratio'] = 0.2
	config['total_ratio'] = 1
	config['supervised_train'] = 'supervised_train'
	config['unsupervised_train'] = 'unsupervised_train'
	assert config['total_ratio'] > config['supervised_ratio']
	return config 


def get_multihot_rnn_dictionary_TF_truven_config():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 10   
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = int(9e5)   ### 3e4  
	### batch_size=8 => 7000 iter <=> 1 epoch 
	config['attention_size'] = 50

	config['data_folder'] = './data'

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'truven_prototype.npy')
	config['prototype_text'] = os.path.join(config['result_folder'], 'truven_prototype_topk')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'truven_5w')
	config['test_file'] = os.path.join(config['data_folder'], 'truven_2w')
	config['mapfile'] = os.path.join(config['data_folder'], 'truven_code2idx')

	lines = open(config['mapfile'], 'r').readlines()
	config['input_dim'] = len(lines)
	assert len(lines) == 283
	config['num_class'] = config['input_dim']
	config['topk'] = 30 

	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ';'
	config['separate_symbol'] = '\t'
	return config 

def semisupervised_get_multihot_rnn_dictionary_TF_truven_config():
	config = get_multihot_rnn_dictionary_TF_truven_config()
	config['supervised_ratio'] = 0.2
	config['total_ratio'] = 1.0
	config['supervised_train'] = 'supervised_train'
	config['unsupervised_train'] = 'unsupervised_train'
	assert config['total_ratio'] > config['supervised_ratio']
	return config


def get_dictionary_TF_truven_config_reconstruction():
	config = {}
	config['batch_size'] = 8
	config['max_length'] = 10   
	config['rnn_in_dim'] = 50
	config['rnn_out_dim'] = 50
	config['rnn_layer'] = 1
	config['batch_first'] = True
	config['LR'] = 1e-1
	config['test_num'] = 3358 
	config['train_iter'] = 2 * 60000 #int(3e3)   ### 3e4  9e5    50000 (N) / 8 (batch_size) * 10 (epoch) = 60000
	### batch_size=8 => 7000 iter <=> 1 epoch 
	config['attention_size'] = 50

	config['data_folder'] = './data'

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'truven_prototype.npy')
	config['prototype_text'] = os.path.join(config['result_folder'], 'truven_prototype_topk')


	config['eta1'] = 1e-3   ## 1e-2	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 0		### classify
	config['lambda1'] = 1e-3 
	config['lambda2'] = 1e-3 ## 1e-2 	
	config['dictionary_size'] = 20

	config['train_file'] = os.path.join(config['data_folder'], 'truven_5w')
	config['test_file'] = os.path.join(config['data_folder'], 'truven_2w')
	config['mapfile'] = os.path.join(config['data_folder'], 'truven_code2idx')

	lines = open(config['mapfile'], 'r').readlines()
	config['input_dim'] = len(lines)
	assert len(lines) == 283
	config['num_class'] = config['input_dim']
	config['topk'] = 30 

	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ';'
	config['separate_symbol'] = '\t'
	return config 


