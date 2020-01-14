#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:00:29 2020

@author: matin
"""

dataset_dir = '/home/matin/Desktop/image_data'
import numpy as np
import cv2
from matplotlib.image import imread
import skimage
from os import listdir
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow 
tensorflow.set_random_seed(42)
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU

img_width=128
img_height=128
counter=0
i=0
dataset = np.ndarray(shape=(707, 3, img_height, img_width),
                     dtype=np.float32)
listimage=[]
listlabel=[]
for ea in listdir(dataset_dir):
    if ea != ".DS_Store":
        counter+=1
        file=dataset_dir+'/'+ea
        
        for fname in listdir(file):
            if fname != ".DS_Store":
                #imread(file+'/'+fname)
                print(file+'/'+fname)
                #x.append(np.array(imread(file+'/'+fname)))
                image = cv2.imread (file+'/'+fname)
                image = skimage.transform.resize(image, (img_width, img_height), mode='reflect')
                listimage.append (image)
                x = img_to_array(image)  
                x = x.reshape((3, 128, 128))
                # Normalize
                x = (x - 128.0) / 128.0
                dataset[i] = x
                i += 1
                listlabel.append(int(counter))

np.random.seed(42)

listlabel  = pd.get_dummies(listlabel, drop_first=False)
X_train, X_test_val, y_train, y_test_val = train_test_split(dataset, listlabel, test_size=0.8, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)



model1=Sequential()
model1.add(Conv2D(6, kernel_size=3, input_shape=(3,img_width, img_height), activation='relu', padding='same'))
model1.add(MaxPool2D(2))
model1.add(Conv2D(12, kernel_size=3, activation='relu', padding='same'))
model1.add(Flatten())
model1.add(Dense(y_train.columns.size, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(X_train, y_train,
          batch_size=500,
          epochs=100,
          verbose=1,
          validation_data=(X_val, y_val))
score = model1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

