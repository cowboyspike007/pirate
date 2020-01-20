#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:40:54 2020

@author: Abhishek
"""

from algorithm.relu2 import resnet
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import cv2
import pickle 
import random

#to save images to background, we will use 'AGG' setting of matplotlib
matplotlib.use('Agg')

#path to dataset
dataset = 'data'

#path to save model
model_path = 'mymodel1.h5'

#path to save labels
label_path = '/'

#path to save plots
plot_path='/'

HP_LR = 1e-3
HP_EPOCHS = 1
HP_IMAGE_DIM = (96,96,3) #
HP_BS = 64
data = []
classes = []

imagepaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagepaths)



for imgpath in imagepaths:
    try:
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (96, 96))
        image_array = img_to_array(image)    
        data.append(image_array)
        label = imgpath.split(os.path.sep)[-2]
        classes.append(label)
    except Exception as e:
        print(e)
        
#print(image_array)

data = np.array(data,dtype='float')/255.0
labels = np.array(classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#print(labels)

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2,random_state=42)
 
aug = ImageDataGenerator(rotation_range=0.25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

classifier = resnet.build(height=96,width=96,depth=3,classes=len(lb.classes_))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum = 0.9, nesterov = True)
opt = Adam(lr=HP_LR, decay=HP_LR/HP_EPOCHS)
classifier.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
csv_logger = CSVLogger('training.log')

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

hist = classifier.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS), validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS,callbacks=[csv_logger,checkpoint])


classifier.save('mymodel1.h5')