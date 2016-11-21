'''
Deep Neural Net for MNIST dataset
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

numNodesHL1 = 500
numNodesHL2 = 500
numNodesHL3 = 500

numClasses = 10
batchSize = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetwork(data):
	hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([784, numNodesHL1])),
	'biases': tf.Variable(tf.random_normal([numNodesHL1]))}
	hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([numNodesHL1, numNodesHL2])),
	'biases': tf.Variable(tf.random_normal([numNodesHL2]))}
	hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([numNodesHL2, numNodesHL3])),
	'biases': tf.Variable(tf.random_normal([numNodesHL3]))}
	outputLayer = {'weights': tf.Variable(tf.random_normal([numNodesHL3, numClasses])),
	'biases': tf.Variable(tf.random_normal([numClasses]))}

	layer1 = tf.add(tf.matmul(data, hiddenLayer1['weights']), hiddenLayer1['biases'])
	layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(layer1, hiddenLayer2['weights']), hiddenLayer2['biases'])
	layer2 = tf.nn.relu(layer2)

	layer3 = tf.add(tf.matmul(layer2, hiddenLayer3['weights']), hiddenLayer3['biases'])
	layer3 = tf.nn.relu(layer3)

	output = tf.add(tf.matmul(layer3, outputLayer['weights']), outputLayer['biases'])

	return output

def trainNN(x):
	prediction = neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

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





