import numpy as np 


def interpret_mimic():
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config
	config = get_multihot_rnn_dictionary_TF_MIMIC3_config()
	mat = np.load(config['prototype_npy'])
	print(mat.shape)





if __name__ == '__main__':
	interpret_mimic()





