import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
numLines = 100000

def createLexicon(pos, neg):
	lexicon = []
	with open(pos, 'r') as f:
		contents = f.readlines()
		for l in contents[:numLines]:
			allWords = word_tokenize(l)
			lexicon += list(allWords)

	with open(neg, 'r') as f:
		contents = f.readlines()
		for l in contents[:numLines]:
			allWords = word_tokenize(l)
			lexicon += list(allWords)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	wordCounts = Counter(lexicon)
	lexicon2 = []

	for w in wordCounts:
		if 1000 > wordCounts[w] > 50:
			lexicon2.append(w)

	print(len(lexicon2))
	return lexicon2

def sampleHandling(sample, lexicon, classification):
	featureSet = []
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:numLines]:
			currentWords = word_tokenize(l.lower())
			currentWords = [lemmatizer.lemmatize(i) for i in currentWords]
			features = np.zeros(len(lexicon))
			for word in currentWords:
				if word.lower() in lexicon:
					indexValue = lexicon.index(word.lower())
					features[indexValue] += 1

			features = list(features)
			featureSet.append([features, classification])
	return featureSet

def create_feature_sets_and_labels(pos, neg, testSize = 0.1):
	lexicon = createLexicon(pos, neg)
	features = []
	features += sampleHandling('pos.txt', lexicon, [1, 0])
	features += sampleHandling('neg.txt', lexicon, [0, 1])
	random.shuffle(features)
	features = np.array(features)

	testingSize = int(testSize * len(features))

	trainX = list(features[:,0][:-testingSize])
	trainY = list(features[:,1][:-testingSize])
	testX = list(features[:,0][-testingSize:])
	testY = list(features[:,1][-testingSize:])

	return trainX, trainY, testX, testY

if __name__ == '__main__':
	trainX, trainY, testX, testY = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([trainX, trainY, testX, testY], f)





