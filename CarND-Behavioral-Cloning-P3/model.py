import os
import argparse
from math import ceil

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from dataloader import Data

def build_model(args):
	model = Sequential()
	
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((60,20), (0,0))))
	
	model.add(Conv2D(24, (5,5), strides=(2, 2)))
	model.add(Activation('elu'))

	model.add(Conv2D(36, (5,5), strides=(2, 2)))
	model.add(Activation('elu'))

	model.add(Conv2D(48, (5,5), strides=(2, 2)))
	model.add(Activation('elu'))

	model.add(Conv2D(64, (3,3), strides=(1, 1)))
	model.add(Activation('elu'))

	model.add(Conv2D(64, (3,3), strides=(1, 1)))
	model.add(Activation('elu'))
	
	model.add(Flatten())
	model.add(Dropout(0.20))

	model.add(Dense(100))
	model.add(Activation('elu'))

	model.add(Dense(50))
	model.add(Activation('elu'))

	model.add(Dense(10))
	model.add(Activation('elu'))

	model.add(Dense(1))
	
	model.summary()

	return model


def train_val(args):
	model = build_model(args)
	
	batch_size = 16	
	data_dir = 'data'
	data_folders = ['udacity', '1', '2', '3', '4', '5', '6']
	model_dir = 'models'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	data_loader = Data(data_dir, data_folders, batch_size=batch_size)
	train_data, val_data = data_loader.getData()
	train_gen, val_gen = data_loader.getGenerators()

	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_gen,
            steps_per_epoch=ceil(len(train_data)/batch_size),
            validation_data=val_gen,
            validation_steps=ceil(len(val_data)/batch_size),
            epochs=4, verbose=1)

	model.save(os.path.join(model_dir, 'model.h5'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Training')
	args = parser.parse_args()
	train_val(args)