import os

def get_multihot_rnn_config():
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






def get_multihot_dictionary_rnn_config():
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
	config['mapfile'] = os.path.join(config['data_folder'], 'SNOW_vocabMAP.txt')

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'heartfailure_prototoye.npy')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10

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
	config['LR'] = 1e-2
	config['test_num'] = 3358 
	config['train_iter'] = int(3e4)  
	### batch_size=8 => 1680 iter <=> 1 epoch 

	config['data_folder'] = './data'

	config['result_folder'] = './result'
	config['prototype_npy'] = os.path.join(config['result_folder'], 'mimic_prototoye.npy')


	config['eta1'] = 1e0	### dictionary
	config['eta2'] = 1e-3	### reconstruction
	config['eta3'] = 1		### classify
	config['lambda1'] = 1e-3
	config['lambda2'] = 1e-2	
	config['dictionary_size'] = 10

	### MIMIC 3 
	config['train_file'] = os.path.join(config['data_folder'], 'mimic_train')
	config['test_file'] = os.path.join(config['data_folder'], 'mimic_test')
	config['mapfile'] = os.path.join(config['data_folder'], 'code_map.txt')

	config['separate_symbol_in_visit'] = ' '
	config['separate_symbol_between_visit'] = ','
	config['separate_symbol'] = '\t'
	return config 







