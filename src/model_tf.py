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

	def train(self, X, Y_1d, seqlen):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen})
		return loss 

	def evaluate(self, X, seqlen):  #### test
		return self.sess.run([self.outputs_prob], \
			feed_dict = {self.X:X, self.seqlen:seqlen})


class AggregateBase(MultihotRnnBase):

	def _build_placeholder(self):
		self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])
		self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.num_class])

	def _build_rnn(self): 
		'''
			rnn-encoder
		'''
		batch_size = tf.shape(self.X)[0] 
		weight_in = tf.Variable(tf.random_normal(shape = [self.input_dim, self.rnn_hidden_num]))
		bias_in = tf.Variable(tf.zeros(shape = [self.rnn_hidden_num]))
		weight_in_2 = tf.Variable(tf.random_normal(shape = [self.rnn_hidden_num, self.rnn_in_dim]))
		bias_in_2 = tf.Variable(tf.zeros(shape = [self.rnn_in_dim]))
		self.rnn_hidden_output = tf.tanh(tf.matmul(self.X, weight_in) + bias_in)
		self.rnn_outputs = tf.nn.sigmoid(tf.matmul(self.rnn_hidden_output, weight_in_2) + bias_in_2)

	def train(self, X, Y_1d):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d})
		return loss 
	def evaluate(self, X):
		return self.sess.run([self.outputs_prob], \
			feed_dict = {self.X:X})




class Multihot_Rnn_Attention(MultihotRnnBase):
	"""
		single attention; scalar-level
	"""
	def __init__(self, **config):
		self.__dict__.update(config)
		### build model
		self._build()
		### session 
		self._open_session()

	def _build_rnn(self):
		batch_size = tf.shape(self.X)[0] 
		x_ = tf.unstack(value = self.X, num = self.max_length, axis = 1)
		assert len(x_) == self.max_length
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.rnn_in_dim)
		outputs, state = tf.contrib.rnn.static_rnn(inputs = x_, cell = lstm_cell,\
		 dtype = tf.float32, sequence_length = self.seqlen)
		assert len(outputs) == self.max_length
		outputs = tf.stack(outputs, axis = 1)
		#index = tf.range(0, batch_size) * self.max_length + (self.seqlen - 1)
		#self.rnn_outputs = tf.gather(tf.reshape(outputs, [-1, self.rnn_in_dim]), index)
		_, self.rnn_outputs = self._build_attention(outputs, self.attention_size)


	def _build_classify_loss(self):
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.logits))



	@staticmethod
	def _build_attention(inputs, attention_size):
		"""
			input: B, T, D: batch_size, timestep, dimension
			output: B, D
			to do: variable-length? ********
		"""

		#dim = inputs.get_shape()[2] ### tf.shape wrong?
		#dim = tf.shape(inputs)[2]
		dim = inputs.shape[2].value
		attention_weight = tf.Variable(tf.random_normal([dim, attention_size]))
		attention_bias = tf.Variable(tf.random_normal([attention_size]))
		attention_dot = tf.Variable(tf.zeros([attention_size]))

		result_1 = tf.tanh(tf.tensordot(inputs, attention_weight, axes = 1) + attention_bias)  ### B, T, D_    D_ is attention_size
		result_2 = tf.tensordot(result_1, attention_dot, axes = 1)  ### B, T
		alpha = tf.nn.softmax(result_2)  ### B, T;  axis = -1

		alpha_expand = tf.expand_dims(alpha, axis = -1)  ### B, T, 1
		output = tf.multiply(inputs, alpha_expand)  ### B, T, D
		output = tf.reduce_sum(output, axis = 1)	### B, D

		return alpha, output 





class Multihot_Rnn_next_visit(MultihotRnnBase):
	"""
		Truven data, rnn-next-visit
	"""
	def _build_classify_loss(self):
		self.classify_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=self.Y,
			logits=self.logits), 1))


	### build model
	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()   ###  self.rnn_outputs
		### forward: full-connect
		weight = tf.Variable(tf.random_normal(shape = [self.rnn_in_dim, self.num_class]))
		bias = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.rnn_outputs, weight) + bias  ### BS,c 
		self.outputs_prob = tf.nn.sigmoid(self.logits)
		### loss 
		self._build_classify_loss() 
		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def _1dlabel_to_2dlabel(self, batch_label):
		'''
			batch_label is list of list. element is integer
		'''
		batch_size = len(batch_label)
		label_2d = np.zeros((batch_size, self.num_class),dtype = int)
		for i in range(batch_size):
			for j in batch_label:
				label_2d[i,j] = 1 
		return label_2d

	def train(self, X, Y_1d, seqlen):
		Y_2d = self._1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen})
		return loss 		


class Multihot_Rnn_Attention_next_visit(Multihot_Rnn_Attention, Multihot_Rnn_next_visit):
	"""
		Truven data, rnn-next-visit
	"""
	def _build_classify_loss(self):
		Multihot_Rnn_next_visit._build_classify_loss(self)

	def _build_rnn(self):
		Multihot_Rnn_Attention._build_rnn(self)

	### build model
	def _build(self):
		Multihot_Rnn_next_visit._build(self)

	def _1dlabel_to_2dlabel(self, batch_label):
		return Multihot_Rnn_next_visit._1dlabel_to_2dlabel(self, batch_label)

	def train(self, X, Y_1d, seqlen):
		return Multihot_Rnn_next_visit.train(self, X, Y_1d, seqlen)










class Multihot_Rnn_Dictionary(MultihotRnnBase):
	"""
		TF; dictionary 
	"""
	def _build_placeholder(self):
		MultihotRnnBase._build_placeholder(self)
		self.X_recon = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])

		####  generation_prototype_patient
		self.prototype_vec = tf.placeholder(dtype = tf.float32, shape = [None, self.dictionary_size])

	def _build_dictionary(self): 
		### forward
		self.dictionary_matrix = tf.Variable(tf.random_normal(
			shape = [self.rnn_in_dim , self.dictionary_size]), 
			dtype = tf.float32)
		self.DTD = tf.matmul(
						 tf.transpose(self.dictionary_matrix, perm = [1,0]),
						 self.dictionary_matrix)
		self.invDTD = tf.linalg.inv(self.DTD)
		self.invDTDDT = tf.matmul( 
							self.invDTD,
							tf.transpose(self.dictionary_matrix, perm = [1,0])
							)
		self.sparse_code_1 = tf.matmul( self.rnn_outputs,
										tf.transpose(self.invDTDDT, perm = [1,0])
			)  #### batch_size, dictionary_size 
		self.sparse_code = tf.maximum(self.sparse_code_1 - self.lambda1, 0)

		### loss 
		self.dictionary_loss = (tf.norm(self.rnn_outputs - 
								tf.matmul(self.sparse_code, tf.transpose(self.dictionary_matrix, perm = [1,0]))
							))**2 \
							+ self.lambda1 * tf.norm(tensor = self.sparse_code, ord = 1) \
							+ self.lambda2 * (tf.norm(tensor = self.dictionary_matrix))**2

	def _build_reconstruction(self):
		### forward
		self.sparse_code_normalized = tf.nn.l2_normalize(self.sparse_code, axis = 1)
		self.ssf = tf.matmul(self.sparse_code_normalized, tf.transpose(self.dictionary_matrix, perm = [1,0]))
		self.weight_reconstruction = tf.Variable(tf.random_normal(
				shape = [self.rnn_in_dim, self.input_dim], 
				dtype = tf.float32))
		self.bias_reconstruction = tf.Variable(tf.zeros(
				shape = [self.input_dim],
				dtype = tf.float32
			))   
		self.output_recon = tf.matmul(self.ssf, self.weight_reconstruction) + self.bias_reconstruction ### tf.sigmoid() 
		### loss 
		self.reconstruction_loss0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_recon,logits=self.output_recon)
		self.reconstruction_loss1 = tf.reduce_sum(self.reconstruction_loss0, axis = 1)
		self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss1)

	def _build_sparse_code_as_input(self):
		self.sparse_code_normalized = tf.nn.l2_normalize(self.prototype_vec, axis = 1)
		self.ssf = tf.matmul(self.sparse_code_normalized, tf.transpose(self.dictionary_matrix, perm = [1,0]))
		self.weight_reconstruction = tf.Variable(tf.random_normal(
				shape = [self.rnn_in_dim, self.input_dim], 
				dtype = tf.float32))
		self.bias_reconstruction = tf.Variable(tf.zeros(
				shape = [self.input_dim],
				dtype = tf.float32
			))   
		self.output_recon = tf.matmul(self.ssf, self.weight_reconstruction) + self.bias_reconstruction ### tf.sigmoid() 
		### loss 		

	def _build(self):
		### placeholder 
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()

		#### I: dictionary learning module
		self._build_dictionary()


		### II: classify Module 
		self.weight_classify = tf.Variable(tf.random_normal(shape = [self.dictionary_size, self.num_class]))
		self.bias_classify = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.sparse_code, self.weight_classify) + self.bias_classify 
		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		self._build_classify_loss() 

		### III: reconstruction module
		self._build_reconstruction()
		self._build_sparse_code_as_input()

		### total loss 
		self.total_loss = self.eta1 * self.dictionary_loss \
						+ self.eta2 * self.reconstruction_loss\
		 				+ self.eta3 * self.classify_loss

		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.total_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def train(self, X, Y_1d, seqlen, X_recon):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		classify_loss, reconstruction_loss, dictionary_loss, _ = self.sess.run(
			[self.classify_loss, self.reconstruction_loss, self.dictionary_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen, self.X_recon:X_recon})
		return classify_loss, reconstruction_loss, dictionary_loss 

	def generation_prototype_patient(self):
		"""
			[1, 0, 0, 0]   [0, 1, 0, 0]   [0, 0, 1, 0],  [0, 0, 0, 1]
		"""
		prototype_vec = np.eye(self.dictionary_size)
		output = self.sess.run([self.output_recon], 
								feed_dict = {self.prototype_vec:prototype_vec})
		output = output[0]
		return output
	'''
	def evaluate(self, X, seqlen):  #### test
		return self.sess.run([self.outputs_prob, self.output_recon], \
			feed_dict = {self.X:X, self.seqlen:seqlen})
	'''

	def evaluate2(self, X, seqlen):
		return self.sess.run([self.output_recon], \
			feed_dict = {self.X:X, self.seqlen:seqlen})		



class Multihot_Rnn_Dictionary2(Multihot_Rnn_Dictionary):
	"""
		TF; dictionary 
	"""
	def _build_placeholder(self):
		MultihotRnnBase._build_placeholder(self)
		self.X_recon = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])

	def _build_dictionary(self): 
		### forward
		self.dictionary_matrix = tf.Variable(tf.random_normal(
			shape = [self.rnn_in_dim , self.dictionary_size]), 
			dtype = tf.float32)
		self.DTD = tf.matmul(
						 tf.transpose(self.dictionary_matrix, perm = [1,0]),
						 self.dictionary_matrix)
		self.invDTD = tf.linalg.inv(self.DTD)
		self.invDTDDT = tf.matmul( 
							self.invDTD,
							tf.transpose(self.dictionary_matrix, perm = [1,0])
							)
		self.sparse_code_1 = tf.matmul( self.rnn_outputs,
										tf.transpose(self.invDTDDT, perm = [1,0])
			)  #### batch_size, dictionary_size 
		self.sparse_code = tf.maximum(self.sparse_code_1 - self.lambda1, 0)

		### loss 
		self.dictionary_loss = (tf.norm(self.rnn_outputs - 
								tf.matmul(self.sparse_code, tf.transpose(self.dictionary_matrix, perm = [1,0]))
							))**2 \
							+ self.lambda1 * tf.norm(tensor = self.sparse_code, ord = 1) \
							+ self.lambda2 * (tf.norm(tensor = self.dictionary_matrix))**2

	def _build_reconstruction(self):
		### forward
		self.sparse_code_normalized = tf.nn.l2_normalize(self.sparse_code, axis = 1)
		self.ssf = tf.matmul(self.sparse_code_normalized, tf.transpose(self.dictionary_matrix, perm = [1,0]))
		self.weight_reconstruction = tf.Variable(tf.random_normal(
				shape = [self.rnn_in_dim, self.input_dim], 
				dtype = tf.float32))
		self.bias_reconstruction = tf.Variable(tf.zeros(
				shape = [self.input_dim],
				dtype = tf.float32
			))   
		self.output_recon = tf.matmul(self.ssf, self.weight_reconstruction) + self.bias_reconstruction ### tf.sigmoid() 
		### loss 
		self.reconstruction_loss0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_recon,logits=self.output_recon)
		self.reconstruction_loss1 = tf.reduce_sum(self.reconstruction_loss0, axis = 1)
		self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss1)


	def _build(self):
		### placeholder 
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()

		#### I: dictionary learning module
		self._build_dictionary()


		### II: classify Module 
		self.weight_classify = tf.Variable(tf.random_normal(shape = [self.dictionary_size, self.num_class]))
		self.bias_classify = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.sparse_code, self.weight_classify) + self.bias_classify 
		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		self._build_classify_loss() 

		### III: reconstruction module
		self._build_reconstruction()

		### total loss 
		self.total_loss = self.eta1 * self.dictionary_loss \
						+ self.eta2 * self.reconstruction_loss\
		 				+ self.eta3 * self.classify_loss

		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.total_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def train(self, X, Y_1d, seqlen, X_recon):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		classify_loss, reconstruction_loss, dictionary_loss, _, output_recon = self.sess.run(
			[self.classify_loss, self.reconstruction_loss, self.dictionary_loss, self.train_fn, self.output_recon], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen, self.X_recon:X_recon})
		return classify_loss, reconstruction_loss, dictionary_loss, output_recon


	def evaluate2(self, X, seqlen):
		return self.sess.run([self.outputs_prob, self.output_recon], \
			feed_dict = {self.X:X, self.seqlen:seqlen})		





class fidelity_network(MultihotRnnBase):
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

	def _build_placeholder(self):
		self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim])
		self.Y = tf.placeholder(dtype = tf.float32, shape = [None, self.num_class])

	def _build_rnn(self):
		weight_in = tf.Variable(tf.random_normal(shape = [self.input_dim, self.rnn_in_dim]))
		bias_in = tf.Variable(tf.zeros(shape = [self.rnn_in_dim]))
		self.rnn_outputs = tf.nn.tanh(tf.matmul(self.X, weight_in) + bias_in)


	def _build_classify_loss(self):
		self.classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.logits))


	### build model
	def _build(self):
		### placeholder
		self._build_placeholder()
		### forward: rnn


		### I: multilayer
		
		self._build_rnn()
		weight = tf.Variable(tf.random_normal(shape = [self.rnn_in_dim, self.num_class]))
		bias = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.rnn_outputs, weight) + bias 


		### II: single layer
		'''weight3 = tf.Variable(tf.random_normal(shape = [self.input_dim, self.num_class]))
		bias3 = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.X, weight3) + bias3
		'''


		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		### loss 
		self._build_classify_loss() 
		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.classify_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	def train(self, X, Y_1d):
		Y_2d = _1dlabel_to_2dlabel(Y_1d)
		loss, _ = self.sess.run([self.classify_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d})
		return loss 

	def evaluate(self, X):  #### test
		return self.sess.run([self.outputs_prob], \
			feed_dict = {self.X:X})



class Multihot_Rnn_Dictionary_fidelity(Multihot_Rnn_Dictionary):
	def _build_reconstruction(self):
		Multihot_Rnn_Dictionary._build_reconstruction(self)
		self.weight2 = tf.Variable(tf.random_normal(shape = [self.input_dim, self.num_class]))
		self.bias2 = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits2 = tf.matmul(self.output_recon, self.weight2) + self.bias2 
		self.outputs_prob2 = tf.nn.softmax(logits = self.logits2, axis = 1)

		self.classify_loss2 = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=self.Y,
			logits=self.logits2), 1))

	def _build(self):
		### placeholder 
		self._build_placeholder()
		### forward: rnn
		self._build_rnn()

		#### I: dictionary learning module
		self._build_dictionary()


		### II: classify Module 
		self.weight_classify = tf.Variable(tf.random_normal(shape = [self.dictionary_size, self.num_class]))
		self.bias_classify = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.sparse_code, self.weight_classify) + self.bias_classify 
		self.outputs_prob = tf.nn.softmax(logits = self.logits, axis = 1)
		self._build_classify_loss() 

		### III: reconstruction module
		self._build_reconstruction()

		### total loss 
		self.total_loss = self.eta1 * self.dictionary_loss \
						+ self.eta2 * self.reconstruction_loss\
		 				+ self.eta3 * self.classify_loss\
		 				+ self.eta3 * self.classify_loss2

		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.total_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))


class Multihot_dictionary_next_visit(Multihot_Rnn_next_visit, Multihot_Rnn_Dictionary):
	def _build_placeholder(self):
		Multihot_Rnn_Dictionary._build_placeholder(self)


	def _build(self):

		Multihot_Rnn_Dictionary._build_placeholder(self)
		Multihot_Rnn_next_visit._build_rnn(self)


		#### I: dictionary learning module
		Multihot_Rnn_Dictionary._build_dictionary(self)


		### II: classify Module   改这个地方 
		self.weight_classify = tf.Variable(tf.random_normal(shape = [self.dictionary_size, self.num_class]))
		self.bias_classify = tf.Variable(tf.zeros(shape = [self.num_class]))
		self.logits = tf.matmul(self.sparse_code, self.weight_classify) + self.bias_classify 
		self.outputs_prob = tf.nn.sigmoid(self.logits)
	

		self._build_classify_loss()

		### III: reconstruction module
		self._build_reconstruction()
		self._build_sparse_code_as_input()

		### total loss 
		self.total_loss = self.eta1 * self.dictionary_loss \
						+ self.eta2 * self.reconstruction_loss\
		 				+ self.eta3 * self.classify_loss
		### train_op
		self.train_fn = tf.train.GradientDescentOptimizer(learning_rate=self.LR).minimize(self.total_loss)
		#acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))


	def train(self, X, Y_1d, seqlen, X_recon):
		Y_2d = self._1dlabel_to_2dlabel(Y_1d)

		#### sess.run 
		classify_loss, reconstruction_loss, dictionary_loss, _ = self.sess.run(
			[self.classify_loss, self.reconstruction_loss, self.dictionary_loss, self.train_fn], \
			feed_dict = {self.X:X, self.Y:Y_2d, self.seqlen:seqlen, self.X_recon:X_recon})
		return classify_loss, reconstruction_loss, dictionary_loss 



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

