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


def interpret_mimic_weight():
	import matplotlib.pyplot as plt
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_config, get_multihot_rnn_dictionary_TF_config, get_multihot_rnn_dictionary_TF_truven_config
	#config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3

	def sigmoid(x):
		return 1. / (1. + np.exp(-x))

	config = get_multihot_rnn_dictionary_TF_truven_config()  ### heart failure
	mat = np.load(config['prototype_npy'])
	mat = sigmoid(mat)

	fin = open(config['mapfile'], 'r')
	code_map = fin.readlines()
	code_map = [line.rstrip() for line in code_map]
	code_num = len(code_map)
	dictionary_size, _ = mat.shape

	### return the top-k
	topk = 2000
	for i in range(dictionary_size):
		print(i)
		vec = mat[i,:code_num]
		vec.sort()
		lst = list(vec[-topk:])
		plt.plot(lst)
	plt.savefig('a.png')



def interpret_truven():
	from config import get_multihot_rnn_dictionary_TF_truven_config, get_dictionary_TF_truven_config_reconstruction
	#config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3
	## truven get_multihot_rnn_dictionary_TF_truven_config
	#config = get_multihot_rnn_dictionary_TF_truven_config()  
	config = get_dictionary_TF_truven_config_reconstruction()
	mat = np.load(config['prototype_npy'])
	print(mat.shape)

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
		#fout.write('Prototype Patient_' + str(i) + ": ")
		fout.write('\item ' + topk_ele + '\n\n')
	fout.close()	



def interpret_mimic_ccs():
	from config import get_multihot_rnn_dictionary_TF_MIMIC3_ccs_config as config_fn
	#config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3
	## truven get_multihot_rnn_dictionary_TF_truven_config
	config = config_fn()  ### heart failure
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
		fout.write('\item ' + topk_ele + '\n\n')
	fout.close()	


def interpret_truven_weight():
	import matplotlib.pyplot as plt
	from config import get_multihot_rnn_dictionary_TF_truven_config
	#config = get_multihot_rnn_dictionary_TF_MIMIC3_config()	### MIMIC 3

	def sigmoid(x):
		return 1. / (1. + np.exp(-x))

	config = get_multihot_rnn_dictionary_TF_truven_config()  ### heart failure
	mat = np.load(config['prototype_npy'])
	mat = sigmoid(mat)

	fin = open(config['mapfile'], 'r')
	code_map = fin.readlines()
	code_map = [line.rstrip() for line in code_map]
	code_num = len(code_map)
	dictionary_size, _ = mat.shape

	### return the top-k
	topk = 10
	threshold = 0.65
	top_risk = []
	for i in range(dictionary_size):
		vec = mat[i,:code_num]
		vec1 = list(vec)
		#vec = list(map(lambda x:0 if x < threshold else x, vec))
		vec2 = list(filter(lambda x:True if x > threshold else False, vec1))
		num = len(vec2)
		topk_ele = list((-vec).argsort())[:num]

		topk_ele = list((-vec).argsort())[:topk]



		top_risk.extend(topk_ele)
	print(len(set(top_risk)) / len(top_risk))



if __name__ == '__main__':
	#interpret_truven()
	interpret_mimic_ccs()
	#interpret_mimic_weight()
	#interpret_truven_weight()
	#interpret_truven()





