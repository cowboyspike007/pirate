#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:37:20 2020

@author: Abhishek
"""

import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt


import keras.backend as k
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
#
import tensorflow as tf

class resnet:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1
        if (k.image_data_format() == 'channels_first'):
            input_shape = (depth, height, width)
            channel_dim = 1
            
        inputs = Input(shape=input_shape)

       # x=keras.layers.Concatenate()([])

        #input1= keras.layers.Input(shape=(96,96,3))
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',input_shape=input_shape)(inputs)
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Second conv block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Third conv block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Fourth conv block
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # Fifth conv block
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # FC layer
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=0.7)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(rate=0.3)(x)

        # Output layer
        output = Dense(units=1, activation='sigmoid')(x)


        keras.layers.Concatenate(axis=-1)
        # Creating model and compiling
        model = Model(inputs=inputs, outputs=output)
     #   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Callbacks
        checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
        
        return model