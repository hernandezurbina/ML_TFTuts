'''
Deep Neural Net for MNIST dataset
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

numClasses = 10
batchSize = 128
epochs = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keepRate = 0.8
keepProb = tf.placeholder(tf.float32)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutionalNN(x):
	weights = {'wConv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			'wConv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			'wFC': tf.Variable(tf.random_normal([7*7*64, 1024])),
			'out': tf.Variable(tf.random_normal([1024, numClasses]))}

	biases = {'bConv1': tf.Variable(tf.random_normal([32])),
			'bConv2': tf.Variable(tf.random_normal([64])),
			'bFC': tf.Variable(tf.random_normal([1024])),
			'out': tf.Variable(tf.random_normal([numClasses]))}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = tf.nn.relu(conv2d(x, weights['wConv1']) + biases['bConv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['wConv2']) + biases['bConv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['wFC']) + biases['bFC'])

	fc = tf.nn.dropout(fc, keepRate)

	output = tf.matmul(fc, weights['out']) + biases['out']

	return output

def trainNN(x):
	prediction = convolutionalNN(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epochLoss = 0.0
			for _ in range(int(mnist.train.num_examples/batchSize)):
				epochX, epochY = mnist.train.next_batch(batchSize)
				_, c = sess.run([optimizer, cost], feed_dict= {x: epochX, y: epochY})
				epochLoss += c
			print("Epoch ", epoch, "completed out of ", epochs, " Loss: ", epochLoss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

trainNN(x)





