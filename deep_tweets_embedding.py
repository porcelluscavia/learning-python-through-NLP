
# Author 1: Samantha Tureski, Martrikelnummer:4109680



import numpy as np
from keras.layers import Dense, Embedding, Dropout, Activation
# from keras.layers import SimpleRNN
# from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential
# from sklearn import metrics

#from Daniel De Kok's implementation of "train.py" in the second Deep Learning Assignment from his SS 2017 Class
class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.start_idx = 1

    def number(self, value, add_if_absent=True):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + self.start_idx
                self.v2n[value] = n
                self.n2v.append(value)
            else:   
                return 0
        return n

    def value(self, number):
        self.n2v[number - 1]

    def max_number(self):
        return len(self.n2v) + 1


def process_file(file_path):

		everything = []
		text_sentm_pair = []
		#
		input = open(file_path)
		for line in input.read().split("\n"):
			data=line.split("\t")
			# print(data)
			try:
				sentiment = data[3]
				if(sentiment != 'neutral'):
					text = data[1].strip()
					text_sentm_pair.append(text)
					text_sentm_pair.append(sentiment)
					everything.append(text_sentm_pair)
					text_sentm_pair = []

			except IndexError:
				#the devset file had an empty entry?
				continue

		# print(everything)
		return everything


def number_data(data,text_numberer,sentm_numberer, max_tweet_len, train = False):
	numbed_sentmts = []

	#I'll deal with putting the sentiments into a numpy array later
	numbed_tweets = np.zeros(shape=(len(data), max_tweet_len))


	for count, textsentm_pair in enumerate(data):
		sentm_numbed = sentm_numberer.number(textsentm_pair[1], train)
		numbed_sentmts.append(sentm_numbed)

		tweet_len = min(max_tweet_len, len(textsentm_pair[0].split()))

		for word_count, word in enumerate(textsentm_pair[0].split()[:tweet_len]):
			word_numbed = text_numberer.number(word, train)
			numbed_tweets[count, word_count] = word_numbed
	#returns a numpy array and a normal list, which I process in the build_model function
	return (numbed_tweets,numbed_sentmts)




def build_model(train_data, val_data, features_num, labels_num):

	tweets_train_numbered = train_data[0]
	labels_train_numbered = np.asarray(train_data[1])
	labels_train_numbered = labels_train_numbered.reshape(len(train_data[1]), 1)
	
	tweets_val_numbered = val_data[0]
	labels_val_numbered = np.asarray(val_data[1])
	labels_val_numbered = labels_val_numbered.reshape(len(val_data[1]), 1)

	print(tweets_train_numbered.shape[1])

	model = Sequential()
	embedding_size = 100
	model.add(Embedding(input_dim = features_num.max_number(),output_dim = embedding_size,input_length = tweets_train_numbered.shape[1]))
	# model.add(SimpleRNN(embedding_size, dropout = 0.4, recurrent_dropout = 0.1))
	model.add(GRU(embedding_size, dropout = 0.4, recurrent_dropout = 0.1))

	model.add(Dense(labels_num.max_number(), activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	print(tweets_train_numbered.shape)
	print(labels_train_numbered.shape)
	model.fit(tweets_train_numbered, labels_train_numbered, epochs= 10, batch_size = 64, verbose=0)
	score = model.evaluate(tweets_val_numbered, labels_val_numbered)
	print(score)

	#pred = model.predict(tweets_val_numbered)
	# print(metrics.f1_score(labels_val_numbered, np.argmax(pred).astype(np.int32)), average = "macro")

	return model


if '__main__' == __name__:
    import sys

    # do we need the temp path?
    try:
        train_file, val_file = sys.argv[1:]
    except ValueError:
        sys.exit("usage: {} train_file val_file".format(sys.argv[0]))

    train_data = process_file(train_file)
    val_data = process_file(val_file)

    labels_num = Numberer()
    features_num = Numberer()
    max_tweet_len = 50

    # # the validation set is currently a twentieth the size of the training set
    # val_set_size = int(len(data_list)/20)

    # train_data = data_list[:-(val_set_size)]
    # val_data = data_list[-(val_set_size):]

    train_data = number_data(train_data,features_num,labels_num, max_tweet_len, train = True)
    val_data = number_data(val_data,features_num,labels_num, max_tweet_len, train = False)


    build_model(train_data, val_data, features_num, labels_num)




# '''
# score =[0.69065488342176451, 0.8613989637305699], so 86% accuracy
# after being disenchanted by DictVectorizer in the secondassignment, 
# and not receiving good results by putting numbered lists directly into 
# an sklearn linear classifier, we decided to try keras' SimpleRNN, GRU, and LSTM here.
# The GRU gave the best results. Loss did not increase over the course of the 10 epochs, 
# and it learned the training set with high accuracy.

# We used the full-size training and dev sets.
# ''


