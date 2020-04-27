# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:04:30 2019

@author: Devanshu
"""

import load_data
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from sklearn.preprocessing import StandardScaler
import random
import time

# Convolution Model
def convolution_model(trainX, trainy, testX, testy, start_time_lda):
	random.seed(10)
	epochs, batch_size, n_outputs = 10, 16, 3
	instance_num, channels_num = trainX.shape
	
	train_x = np.expand_dims(trainX, axis=2)
	test_x = np.expand_dims(testX, axis=2)
	print("After reshaping Training and Testing Data")
	print(train_x.shape,test_x.shape)
	
	test_label_new = np.expand_dims(testy, axis=1)
	train_label_new = np.expand_dims(trainy, axis=1)
	
	# CNN Classifier Model
	model = Sequential(name="CNN Classifier Model")
	model.add(Conv1D(2,2,activation='relu',input_shape=(2, 1)))
	model.add(Dense(32, activation='relu'))
	model.add(Conv1D(2,1,activation='relu'))
	model.add(Conv1D(2,1,activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax')) #softmax beacuse there are three classes
	print(model.summary())
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	print("Training...")
	model.fit(train_x, train_label_new, epochs=epochs, batch_size=batch_size)
	
	end_train_time_lda = time.time()
	print("Training Time including data pre-processing: ",(end_train_time_lda - start_time_lda) / 60,"minutes")
	
	print("Testing...")
	_, accuracy = model.evaluate(test_x, test_label_new, batch_size=batch_size)
	print("After", epochs,"epochs with",batch_size,"batch size")
	print("Testing Accuracy: ",accuracy)
	

def preprocessing_methods(dataset, dataset_label):
	# Data Augmentation
	ss = StandardScaler()
	dataset = ss.fit_transform(dataset)
	
	# Feature extraction using Linear Discrimination Analysis (LDA)
	lda = LinearDiscriminantAnalysis(n_components=2)
	
	# Implementing LDA feature extraction
	dataset_fe = lda.fit(dataset, dataset_label).transform(dataset)
	
	# Split dataset into train and test after LDA feature extraction
	train_fe = dataset_fe[:84420, :]
	test_fe = dataset_fe[84420:, :]
	print("Original Training and Testing Shapes")
	print(train_fe.shape,test_fe.shape)
	
	return train_fe, test_fe

# Main function
def main():
	# return dataset from load_data function in load_data.py script
	data = load_data.read_data_sets(one_hot=True)
	
	# get train data and labels by batch size 
	# training = 84420 and testing = 58128
	BATCH_SIZE = 84420
	train_x, train_labels = data.train.next_batch(BATCH_SIZE)
	
	# get test data
	test_x = data.test.data
	
	# get test labels
	test_labels = data.test.labels
	
	# to get class ratio in the testing dataset
	total_count = 0
	pos_count = 0
	neg_count = 0
	neu_count = 0
	
	#Label formation
	train_label = []
	for i in range(len(train_labels)):
		total_count = total_count + 1
		if train_labels[i,0] == 1:
			# If emotion is Positive set label to 1
			train_label.append(1)
			pos_count = pos_count + 1
		if train_labels[i,1] == 1:
			# If emotion is neutral set label to 0
			train_label.append(0)
			neu_count = neu_count + 1
		if train_labels[i,2] == 1:
			# If emotion is negative set label to -1
			train_label.append(-1)
			neg_count = neg_count + 1
	
	test_label = []
	for i in range(len(test_labels)):
		total_count = total_count + 1
		if test_labels[i,0] == 1:
			# If emotion is Positive set label to 1
			test_label.append(1)
			pos_count = pos_count + 1
		if test_labels[i,1] == 1:
			# If emotion is neutral set label to 0
			test_label.append(0)
			neu_count = neu_count + 1
		if test_labels[i,2] == 1:
			# If emotion is negative set label to -1
			test_label.append(-1)
			neg_count = neg_count + 1
	
	# Merge train and test dataset
	train_np = np.array(train_x)
	test_np = np.array(test_x)
	dataset = np.concatenate((train_np, test_np))
	
	# Merge train and test labels
	train_label_np = np.array(train_label)
	test_label_np = np.array(test_label)
	dataset_label = np.concatenate((train_label_np, test_label_np))
	
	# Starting timer
	start_time_lda = time.time()
	
	# Implementing pre processing steps such as data augmentation and feature extraction
	train_fe, test_fe = preprocessing_methods(dataset, dataset_label)
	
	# Implementing CNN model for classification
	convolution_model(train_fe, train_labels, test_fe, test_labels, start_time_lda)
	

if __name__ == '__main__':
	main()