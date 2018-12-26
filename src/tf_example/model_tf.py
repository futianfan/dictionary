import numpy as np
import tensorflow as tf
tf.set_random_seed(3)


## 1. hyperparameters
from stream import max_length, batch_size, TrainFile, TestFile, Create_Multihot_Data
from stream import admis_dim as input_dim
from stream	import batch_size as global_batch_size
lr = 0.001
training_iters = 100000
n_hidden_units = 100   # neurons in hidden layer
n_classes = 2      


## 2. placeholder
x = tf.placeholder(tf.float32, [None, max_length, input_dim])
y = tf.placeholder(tf.float32, [None, n_classes])
x_len = tf.placeholder(tf.float32, [None])
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([input_dim, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, X_len, batch_size, weights, biases):
    # hidden layer for input to cell
    ########################################
    X = tf.reshape(X, [-1, input_dim])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, max_length, n_hidden_units])

    # cell
    ##########################################
    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell = cell, inputs = X_in, sequence_length = X_len, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # # or
    # unpack to list [(batch, outputs)..] * steps
    print(outputs.shape)
    results = tf.matmul(outputs[ np.arange(batch_size), -1, :], weights['out']) + biases['out']    # shape = (128, 10)
    print('ok')
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
        #outputs = tf.unpack(outputs)
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        #outputs = tf.unstack(outputs)
    #results = tf.matmul(outputs[np.arange(batch_size), -1], weights['out']) + biases['out']    # shape = (128, 2)
    print(outputs[0].shape)
    results = tf.matmul(outputs[X_len, np.arange(batch_size), :], weights['out']) + biases['out']    # shape = (128, 10)
    return results

#x_len = list(x_len.reshape(-1))
#x_len = list(tf.reshape(x_len, [-1]))
pred = RNN(x, x_len, batch_size, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('finish construction of computational graph')




trainData = Create_Multihot_Data(TrainFile, batch_size = global_batch_size)

with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init) 
    step = 0
    while step * global_batch_size < training_iters:
        ##batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs, batch_xs_len, batch_ys = trainData.next()
        bs = len(batch_xs_len)
        #batch_xs = batch_xs.reshape([batch_size, max_length, input_dim])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            x_len: batch_xs_len,
            batch_size: bs, 
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            x_len: batch_xs_len,
            batch_size: bs,  
            y: batch_ys,
            }))
        step += 1


"""

def full_connected_layer(inputs, in_size, out_size, activation_function=None,):
	# add one more layer and return the output of this layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b,)
	return outputs

'''
class OnehotRNN(object):
	def __init__(self, **config):
		### 1. hyperparameter
		self.max_length = config['max_length']
		self.admis_num = config['admis_num']
		self.class_num = config['class_num']

		### 2. placeholder
		self.input_x = tf.placeholder(tf.int32, [None, self.max_length, self.admis_num], name = 'input_x')
		self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name = 'input_y')

		### 3.  feedforward
'''

## Suppoose given X, X_lengths, 
max_length = 5
admis_num = 1900
class_num = 2

input_x = tf.placeholder(tf.int32, [None, self.max_length, self.admis_num], name = 'input_x')
input_y = tf.placeholder(tf.float32, [None, self.class_num], name = 'input_y')
cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
outputs, last_states = tf.nn.dynamic_rnn(
		cell=cell,
		dtype=tf.float64,
		sequence_length=x_lengths,
		inputs=input_x)
result = tf.contrib.learn.run_n(
		{"outputs": outputs, "last_states": last_states},
		n=1,
		feed_dict=None)



if __name__ == '__main__':

	X = np.random.randn(2, 10, 8)
	X[1,6:] = 0
	X_lengths = [10, 6]
	cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
	outputs, last_states = tf.nn.dynamic_rnn(
		cell=cell,
		dtype=tf.float64,
		sequence_length=X_lengths,
		inputs=X)
	result = tf.contrib.learn.run_n(
		{"outputs": outputs, "last_states": last_states},
		n=1,
		feed_dict=None)
	assert result[0]["outputs"].shape == (2, 10, 64)
	assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()
	assert result[0]['last_states'][0].shape == (2,64)
	assert result[0]['last_states'][1].shape == (2,64)
"""

