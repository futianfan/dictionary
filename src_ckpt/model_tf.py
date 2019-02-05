import numpy as np 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__all__ = [
	'MultihotRnnBase',
	'Multihot_Rnn_Dictionary',
]
tf.set_random_seed(5)   ### 3, 


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
		#self._open_session()


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
		self.rnn_outputs = tf.gather(tf.reshape(outputs, [-1, self.rnn_in_dim]), index)

	def _build_classify_loss(self):
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.logits))


	### build model
	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()
		### forward: full-connect
		weight = tf.Variable(tf.random_normal(shape = [self.rnn_in_dim, self.num_class]))
		bias = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.rnn_outputs, weight) + bias 
		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		### loss 
		self._build_classify_loss() 
		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def train(self, X, Y_1d, seqlen, sess):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen})
		return loss 

	def evaluate(self, X, seqlen, sess):  #### test
		return sess.run([self.outputs_prob], \
			feed_dict = {self.X:X, self.seqlen:seqlen})
	'''
	def evaluate_loss(self, X, Y_1d, seqlen, sess):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss = sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen})
		loss = loss[0]
		return loss 
	'''

if __name__ == '__main__':
	'''
	from config import get_multihot_rnn_TF_config
	config = get_multihot_rnn_TF_config()
	multihot_rnn_base = MultihotRnnBase(**config)
	'''

	from config import get_multihot_rnn_dictionary_TF_config
	config = get_multihot_rnn_dictionary_TF_config()
	multihot_rnn_dictionary = Multihot_Rnn_Dictionary(**config)

