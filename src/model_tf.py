import numpy as np 
import tensorflow as tf

__all__ = [
	'MultihotRnnBase',
]


def _1dlabel_to_2dlabel(batch_label):
	batch_size = len(batch_label)
	label_2d = np.zeros((batch_size, 2),dtype = int)
	for i in range(batch_size):
		label_2d[i, int(batch_label[i])] = 1
	return label_2d

class MultihotRnnBase(object):
	"""
	input is multihot label vector	
	"""

	def __init__(self, **config):
		### hyperparameter list
		'''self.max_length = config['max_length']
		self.batch_size = config['batch_size']
		self.input_dim = config['input_dim']
		self.rnn_in_dim = config['rnn_in_dim']
		self.rnn_out_dim = config['rnn_out_dim']
		self.num_class = config['num_class']
		self.LR = config['LR']'''
		self.__dict__.update(config)
		### build model
		self._build()
		### session 
		self._open_session()


	def _open_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_placeholder(self):
		self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.max_length, self.input_dim])
		self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.num_class])
		self.seqlen = tf.placeholder(dtype = tf.int32, shape = [None])

	def _build_rnn(self):
		batch_size = tf.shape(self.X)[0] 
		x_ = tf.unstack(value = self.X, num = self.max_length, axis = 1)
		assert len(x_) == self.max_length
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.rnn_in_dim)
		outputs, state = tf.contrib.rnn.static_rnn(inputs = x_, cell = lstm_cell,\
		 dtype = tf.float32, sequence_length = self.seqlen)
		assert len(outputs) == self.max_length
		outputs = tf.stack(outputs, axis = 1)
		index = tf.range(0, batch_size) * self.max_length + (self.seqlen - 1)
		self.outputs = tf.gather(tf.reshape(outputs, [-1, self.rnn_in_dim]), index)

	def _build_classify_loss(self):
		self.cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.logits))


	### build model
	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()
		### forward: full-connect
		weight = tf.Variable(tf.random_normal(shape = [self.rnn_in_dim, self.num_class]))
		bias = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.outputs, weight) + bias 
		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		### loss 
		self._build_classify_loss() 
		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.cost_fn)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def train(self, X, Y_1d, seqlen):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.cost_fn, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen})
		return loss 

	def evaluate(self, X, seqlen):  #### test
		return self.sess.run([self.outputs_prob], \
			feed_dict = {self.X:X, self.seqlen:seqlen})



class Multihot_Rnn_Dictionary(MultihotRnnBase):
	"""

	"""
	def _build_placeholder(self):
		MultihotRnnBase._build_placeholder(self)
		self.X_recon = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])

	def _build_dictionary(self):
		pass 
		### forward

		### loss
		self.dictionary_loss = 0

	def _build_reconstruction(self):
		### forward
		pass 
		### loss 


	def _build(self):
		### placeholder 
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()

		#### I: dictionary learning module

		self._build_dictionary()


		### II: classify Module
		"""
		xxx
		""" 
		self._build_classify_loss() 

		### III: reconstruction module



		### train_op
		#self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.cost_fn)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))







if __name__ == '__main__':
	'''
	from config import get_multihot_rnn_TF_config
	config = get_multihot_rnn_TF_config()
	multihot_rnn_base = MultihotRnnBase(**config)
	'''

	from config import get_multihot_rnn_dictionary_TF_config
	config = get_multihot_rnn_dictionary_TF_config()
	multihot_rnn_dictionary = Multihot_Rnn_Dictionary(**config)


"""
	WEIGHTS={'out': tf.Variable(tf.random_normal(shape=[rnn_hidden_size, nums_classes]))}
	BIAS = {'out': tf.Variable(tf.random_normal(shape=[nums_classes]))}

	def dynamic_rnn(x, seqlen, weights, bias):
		x_input = tf.unstack(value=x, num=max_seq_len, axis=1)
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size)
		outputs, state = tf.contrib.rnn.static_rnn(inputs=x_input, cell=lstm_cell, dtype=tf.float32, sequence_length=seqlen)
		outputs = tf.stack(outputs, axis=1)
		batch_size = tf.shape(outputs)[0]
		index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
		outputs = tf.gather(tf.reshape(outputs, [-1, rnn_hidden_size]), index)  # tf.gather根据序号默认抽取行
		return tf.matmul(outputs, weights) + bias

	logits = dynamic_rnn(x=X, seqlen=seqlen, weights=WEIGHTS['out'], bias=BIAS['out'])
	y_pred = tf.nn.softmax(logits=logits)




"""





"""
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
"""

