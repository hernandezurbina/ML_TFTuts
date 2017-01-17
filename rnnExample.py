'''
Deep Neural Net for MNIST dataset
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

epochs = 3
numClasses = 10
batchSize = 128

chunkSize = 28
numChunks = 28

rnnSize = 128

x = tf.placeholder('float', [None, numChunks, chunkSize])
y = tf.placeholder('float')

def recurrentNN(x):
	layer = {'weights': tf.Variable(tf.random_normal([rnnSize, numClasses])),
			'biases': tf.Variable(tf.random_normal([numClasses]))}

	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, chunkSize])
	x = tf.split(0, numChunks, x)

	lstmCell = rnn_cell.BasicLSTMCell(rnnSize)
	outputs, states = rnn.rnn(lstmCell, x, dtype=tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

	return output

def trainNN(x):
	prediction = recurrentNN(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)	

	with tf.Session() as sess:
		sess.run(tf.tf.global_variables_initializer())

		for epoch in range(epochs):
			epochLoss = 0.0
			for _ in range(int(mnist.train.num_examples/batchSize)):
				epochX, epochY = mnist.train.next_batch(batchSize)
				epochX = epochX.reshape((batchSize, numChunks, chunkSize))
				_, c = sess.run([optimizer, cost], feed_dict= {x: epochX, y: epochY})
				epochLoss += c
			print("Epoch ", epoch, "completed out of ", epochs, " Loss: ", epochLoss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ", accuracy.eval({x: mnist.test.images.reshape((-1, numChunks, chunkSize)), 
			y: mnist.test.labels}))

trainNN(x)





