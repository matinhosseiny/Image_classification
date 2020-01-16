# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:15:53 2019

@author: matin
"""
dataset_dir = 'image_data'
import numpy as np
import cv2
from matplotlib.image import imread
from keras.models import Sequential, Model 
import skimage
from os import listdir
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_width=256
img_height=256
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
                x = x.reshape((3, 256, 256))
                # Normalize 
                x = (x) / 256
                dataset[i] = x
                i += 1
                listlabel.append(int(counter))

import tensorflow 

from sklearn.model_selection import train_test_split
import pandas as pd
listlabel  = pd.get_dummies(listlabel, drop_first=False)
X_train, X_test_val, y_train, y_test_val = train_test_split(dataset, listlabel, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
from keras import optimizers

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization,LeakyReLU
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0],  256, 256,3)
X_test = X_test.reshape(X_test.shape[0],  256, 256,3)
X_val = X_val.reshape(X_val.shape[0],256, 256,3)
from keras import applications
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
for layer in model.layers[:6]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(9, activation="softmax")(x)
model1 = Model(input = model.input, output = predictions)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train, y_train,
          batch_size=20,
          epochs=100,
          verbose=1,
          validation_data=(X_val, y_val))
score = model1.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])