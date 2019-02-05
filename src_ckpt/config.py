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
	config['LR'] = 1e-3
	config['test_num'] = 3358 
	config['train_iter'] = int(2e6)

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')

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
	config['LR'] = 1e-4
	config['test_num'] = 3358 
	config['train_iter'] = int(2e6)
	config['valid_iter'] = 10 

	config['data_folder'] = './data'
	config['train_file'] = os.path.join(config['data_folder'], 'training_data_1.txt')
	config['test_file'] = os.path.join(config['data_folder'], 'test_data_1.txt')
	config['log_folder'] = 'log'
	config['log_train_folder'] = os.path.join(config['log_folder'], 'train')
	config['log_valid_folder'] = os.path.join(config['log_folder'], 'valid')
	config['bestmodel_save_path'] = os.path.join(config['log_valid_folder'], 'bestmodel')

	return config 


