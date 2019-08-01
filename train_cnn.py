#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:03:07 2019

@author: stevencheung
"""

import os, shutil
base_dir = './mydata'

train_dir = os.path.join(base_dir, 'training_set')
val_dir = os.path.join(base_dir, 'test_set')

    
from keras.layers import Dropout    
from keras import layers
from keras import models
from keras.applications import VGG16
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
conv_base = VGG16(weights='imagenet',
                  include_top=False, #we are going to remove the top layer, VGG was trained for 1000 classes, here we only have two
                  input_shape=(224, 224, 3))
conv_base.trainable = False

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(layers.Dense(26, activation='softmax'))
model.summary()


from keras import optimizers
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')


print ("start..")
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
