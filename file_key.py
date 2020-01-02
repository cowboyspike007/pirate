#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:21:48 2019

@author: Abhishek
"""
from os import listdir
from PIL import Image as PImage
reader=open('train_labeled_studies.csv')
dir=[]
value={}
com='your dir address to the MURA folder' #'/Users/apple/Desktop/'
for line in reader:
    (com1, val) = line.split(',')
    key=com+com1    
    dir.append(key)
    value[str(key)] = int(val)
def loadImages(path):
    # return array of images
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)
    return loadedImages
# your images in an array
for path in dir:
    imgs = loadImages(path)
    for img in imgs:
    # you can show every image
        img.show()