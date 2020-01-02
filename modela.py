#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:28:35 2019

@author: Abhishek
"""
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D

class modela:
        def build(height,width,depth,classes):
            model=Sequential()
            input_shape=(height,width,depth)
            channel_dim = -1
            if(k.image_data_format() == 'channels_first'):
                input_shape = (depth,width,height)
                channel_dim = 1

            model.add(Conv2D(32, (3,3), activation='relu',input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3,3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3,3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (3,3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3,3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3,3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))

            Flatten()

            Dense(256, activation='relu')
            Dropout(0.5)

            Dense(128, activation='relu')
            Dropout(0.3)

            Dense(64, activation='relu')
            Dropout(0.2)

            Dense(1, activation='sigmoid')

            return model
