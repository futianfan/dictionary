import tensorflow as tf
import numpy as np 

np.random.seed(1)


def _1dlabel_to_2dlabel(batch_label):
	batch_size = len(batch_label)
	label_2d = np.zeros((batch_size, 2),dtype = int)
	for i in range(batch_size):
		label_2d[i, int(batch_label[i])] = 1
	return label_2d

def train():
	from stream import Create_Multihot_Data
	from config import get_multihot_rnn_TF_config

	config = get_multihot_rnn_TF_config()	
	train_iter = config['train_iter']
	LR = config['LR']
	TrainData = Create_Multihot_Data(is_train = True, **config)
	TestData = Create_Multihot_Data(is_train = False, **config)

	### model hyperparameter
	max_seq_len = config['max_length']
	input_dim = config['input_dim']
	nums_classes = config['num_class']
	rnn_hidden_size = config['rnn_out_dim']

	#### construct graph
	X = tf.placeholder(dtype=tf.float32, shape=[None, max_seq_len, input_dim])
	Y = tf.placeholder(dtype=tf.float32, shape=[None,nums_classes])
	seqlen = tf.placeholder(dtype=tf.int32, shape=[None])

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

	cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))
	train_fn = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(cost_fn)
	acc_fn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1)), tf.float32))

	### train: session 
	print('run session')
	init_parm = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_parm)
		epoch = 0
		total_loss = 0
		for i in range(train_iter):
			if i % TrainData.num_of_iter_in_a_epoch == 0 and i > 0:
				epoch += 1
				print('{} epoch, loss:{}; '.format(epoch, total_loss))
				total_loss = 0
			data, data_len, label = TrainData.next()
			label = _1dlabel_to_2dlabel(label)
			loss, _ = sess.run([cost_fn, train_fn], feed_dict = {X:data, Y:label, seqlen:data_len})
			total_loss += loss 


	'''
    with tf.Session() as sess:
        sess.run(init_parm)
        for iter in range(1, nums_training+1):
            batch_data, batch_labels, batch_seq_len = train_sets.next(batch_size=batch_size)
            # x_batch = np.reshape(batch_data, [batch_size, max_seq_len, 1])
            loss, _ = sess.run([cost_fn, train_fn], feed_dict={X:batch_data, Y:batch_labels, seqlen:batch_seq_len})
            if iter % nums_display == 0 or iter ==1:
                accuracy = sess.run(acc_fn, feed_dict={X:batch_data, Y:batch_labels,seqlen:batch_seq_len})
                print("Step:%d, loss=%.4f, accuracy=%.4f" % (iter, loss, accuracy))
        print("train finished!")
        test_accuracy = sess.run(acc_fn, feed_dict={X:test_sets.data, Y:test_sets.labels, seqlen:test_sets.seq_len})
        print("test accuracy: %.4f" % test_accuracy)
	'''



if __name__ == '__main__':
	train()



