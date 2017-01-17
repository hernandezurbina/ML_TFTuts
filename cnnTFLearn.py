import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

convNet = input_data(shape=[None, 28, 28, 1], name='input')
convNet = conv_2d(convNet, 32, 2, activation='relu')
convNet = max_pool_2d(convNet, 2)

convNet = conv_2d(convNet, 64, 2, activation='relu')
convNet = max_pool_2d(convNet, 2)

convNet = fully_connected(convNet, 1024, activation='relu')
convNet = dropout(convNet, 0.8)

convNet = fully_connected(convNet, 10, activation='softmax')
convNet = regression(convNet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convNet)

model.fit({'input': X}, {'targets': Y}, n_epoch=10, 
	validation_set=({'input': testX}, {'targets': testY}),
	snapshot_step=500, show_metric=True, run_id='mnist')

model.save('tfLearnCNN.model')

# model.load('tfLearnCNN.model')
# print(model.predict([testX[1]]))
