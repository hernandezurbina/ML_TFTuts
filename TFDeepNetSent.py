import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np

trainX, trainY, testX, testY = create_feature_sets_and_labels('pos.txt','neg.txt')

numNodesHL1 = 1500
numNodesHL2 = 1500
numNodesHL3 = 1500

numClasses = 2
batchSize = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

hiddenLayer1 = {'f_fum': numNodesHL1,
				'weight': tf.Variable(tf.random_normal([len(trainX[0]), numNodesHL1])),
				'bias': tf.Variable(tf.random_normal([numNodesHL1]))}
hiddenLayer2 = {'f_fum': numNodesHL2,
				'weight': tf.Variable(tf.random_normal([numNodesHL1, numNodesHL2])),
				'bias': tf.Variable(tf.random_normal([numNodesHL2]))}
hiddenLayer3 = {'f_fum': numNodesHL3,
				'weight': tf.Variable(tf.random_normal([numNodesHL2, numNodesHL3])),
				'bias': tf.Variable(tf.random_normal([numNodesHL3]))}
outputLayer = {'f_fum': None,
				'weight': tf.Variable(tf.random_normal([numNodesHL3, numClasses])),
				'bias': tf.Variable(tf.random_normal([numClasses]))}



def neuralNetwork(data):
	layer1 = tf.add(tf.matmul(data, hiddenLayer1['weight']), hiddenLayer1['bias'])
	layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(layer1, hiddenLayer2['weight']), hiddenLayer2['bias'])
	layer2 = tf.nn.relu(layer2)

	layer3 = tf.add(tf.matmul(layer2, hiddenLayer3['weight']), hiddenLayer3['bias'])
	layer3 = tf.nn.relu(layer3)

	output = tf.add(tf.matmul(layer3, outputLayer['weight']), outputLayer['bias'])

	return output

def trainNN(x):
	prediction = neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	epochs = 10

	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epochLoss = 0.0
			i = 0
			while i < len(trainX):
				start = i
				end = i + batchSize
				batchX = np.array(trainX[start:end])
				batchY = np.array(trainY[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})

				epochLoss += c
				i += batchSize

			print("Epoch ", epoch + 1, " completed out of ", epochs, " Loss: ", epochLoss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print("Accuracy: ", accuracy.eval({x: testX, y: testY}))

trainNN(x)





