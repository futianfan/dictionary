import numpy as np 


def interpret_mimic():
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config, get_multihot_rnn_dictionary_TF_config
	config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3
	#config = get_multihot_rnn_dictionary_TF_config()  ### heart failure
	mat = np.load(config['prototype_npy'])

	fin = open(config['mapfile'], 'r')
	code_map = fin.readlines()
	code_map = [line.rstrip() for line in code_map]

	assert len(code_map) == mat.shape[1]




if __name__ == '__main__':
	interpret_mimic()





