import numpy as np 


def interpret_mimic():
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config, get_multihot_rnn_dictionary_TF_config
	#config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3
	config = get_multihot_rnn_dictionary_TF_config()  ### heart failure
	mat = np.load(config['prototype_npy'])

	fin = open(config['mapfile'], 'r')
	code_map = fin.readlines()
	code_map = [line.rstrip() for line in code_map]
	code_num = len(code_map)
	dictionary_size, _ = mat.shape

	### return the top-k
	topk = 10
	fout = open(config['prototype_text'], 'w')
	for i in range(dictionary_size):
		vec = mat[i,:code_num]
		topk_ele = list((-vec).argsort())[:topk]
		topk_ele = [code_map[i].capitalize() for i in topk_ele]
		topk_ele = '; '.join(topk_ele)
		fout.write('Prototype Patient_' + str(i) + ": ")
		fout.write(topk_ele + '\n')
	fout.close()


if __name__ == '__main__':
	interpret_mimic()





