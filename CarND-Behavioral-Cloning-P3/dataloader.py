import os
import csv
import random
from random import shuffle

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Data(object):

	def __init__(self , data_dir, data_folders, batch_size=32):
		self.__data_dir = data_dir
		csv_data = []
		for folder in data_folders:
			with open(os.path.join(data_dir, folder, 'driving_log.csv') , 'r') as f:
				reader = csv.reader(f)
				for i,line in enumerate(reader):
					if i==0:
						continue
					name_parts = line[0].split('/')
					if len(name_parts) == 2:
						line[0] = os.path.join('udacity/' , line[0])
						line[1] = os.path.join('udacity/' , line[0])
						line[2] = os.path.join('udacity/' , line[0])
					csv_data.append(line)
		
		self.train_data, self.val_data = train_test_split(csv_data, test_size=0.2)
		self.__batch_size = batch_size

	def generator(self, samples, batch_size):
		num_samples = len(samples)
		while 1: # Loop forever so the generator never terminates
			shuffle(samples)
			for offset in range(0, num_samples, batch_size):
				batch_samples = samples[offset:offset+batch_size]

				images = []
				angles = []
				for batch_sample in batch_samples:
					# Center
					name_parts = batch_sample[0].split('/')
					name = name_parts[-3] + '/' + name_parts[-2] + '/' + name_parts[-1]
					name = os.path.join(self.__data_dir, name)
					center_image = cv2.imread(name)
					center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
					center_angle = float(batch_sample[3])
					images.append(center_image)
					angles.append(center_angle)
					images.append(np.fliplr(center_image))
					angles.append(-1.0*center_angle)

					correction = 0.2
					#Left
					left_angle = center_angle + correction
					name_parts = batch_sample[1].split('/')
					name = name_parts[-3] + '/' + name_parts[-2] + '/' + name_parts[-1]
					name = os.path.join(self.__data_dir, name)
					left_image = cv2.imread(name)
					left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
					images.append(left_image)
					angles.append(left_angle)
					images.append(np.fliplr(left_image))
					angles.append(-1.0*left_angle)

					#Right
					right_angle = center_angle - correction
					name_parts = batch_sample[2].split('/')
					name = name_parts[-3] + '/' + name_parts[-2] + '/' + name_parts[-1]
					name = os.path.join(self.__data_dir, name)
					right_image = cv2.imread(name)
					right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
					images.append(right_image)
					angles.append(right_angle)
					images.append(np.fliplr(right_image))
					angles.append(-1.0*right_angle)

				X_train = np.array(images)
				y_train = np.array(angles)
				yield sklearn.utils.shuffle(X_train, y_train)

	def getData(self):
		return self.train_data, self.val_data

	def getGenerators(self):
		self.__train_gen = self.generator(self.train_data, self.__batch_size)
		self.__val_gen = self.generator(self.val_data, self.__batch_size)
		return self.__train_gen, self.__val_gen

if __name__ == '__main__':
	data_loader = Data('data', ['1'])
	train_gen, val_gen = data_loader.getGenerators()
	train_sample = next(train_gen)
	num_columns = 5
	num_rows = 5
	f, axarr = plt.subplots(num_rows , num_columns, figsize=(12,10), squeeze=False)
	for i in range(25):
		row = int(i/num_columns)
		column = i%num_columns
		img = train_sample[1][i]
		axarr[row][column].axis('off')
		axarr[row][column].imshow(train_sample[0][i])
		axarr[row][column].set_title("{:.4f}".format(train_sample[1][i]))
	f.suptitle("Sample Images with Steering angles")
	plt.show()