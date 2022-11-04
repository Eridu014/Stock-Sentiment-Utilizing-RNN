
from keras import  models
from keras import backend as k
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras import layers
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Embedding, Flatten, Dense, SimpleRNN
#Question 1. Load the file named "stock_sentiments.csv". The file contains titles of news articles and their corresponding sentiment scores. Review teh file before proceeding. (10 points)

dataset = pd.read_csv("stock_sentiments.csv", skiprows=1)
print(dataset.head())

print(dataset.describe())
#Question 2. Create a tokenizer and remove all non-alphabetic characters from the text. Fit the tokenizer to the data. Create a word index. The maximum number of words is 1000. (30 points)
#create tokenizer
tokenizer = Tokenizer(num_words = 1000, filters = '0123456789!"#$%&()*+,-/.:;<>?@[]\|^_,~', lower = True)
#create word index
tokenizer.fit_on_texts(dataset)




#Question 3. Convert the text to sequences and pad the sequences to the maximum length of 50.(10 points)
#turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(dataset)
print(sequences[1])

maxlen = 20

one_hot_results = tokenizer.texts_to_matrix(dataset, mode = 'binary')
print(one_hot_results.shape)
#word index
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

print(word_index)
#Question 4. Convert the sentiment values to categorical. (10 points)
#Question 5. Split the data into 90% training and 10% testing. (10 points)
(input_train, y_train), (input_test, y_test) = dataset

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

input_train = sequences.pad_sequences(input_train, maxlen = maxlen)
input_test = sequences.pad_sequences(input_test, maxlen = maxlen)
#Question 6. Create a Sequential model with embedding, LSTM, Dropout, and Dense layers. The embedding dimension is 50. specify appropriate parameters. Compile and fit the model. (20 points)
#words as features
max_features = 10000
#top max feature's most common words

model_lstm = models.Sequential()
model_lstm.add(Embedding(1000, 8, input_length = maxlen))
model_lstm.add(LSTM(32))
model_lstm.add(Flatten())
#classifier
model_lstm.add(Dense(units = 1), activation = 'sigmoid')

#Question 7. Clear the session and create a GRU model. Compile and fit the model. (10 points)

model_lstm.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model_lstm.summary()

history = model_lstm.fit(input_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.1)
