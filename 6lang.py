import os
from turtle import color
import cv2
import numpy as np
from sklearn.feature_extraction import image
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.applications import VGG19
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import h5py

BASE_DIR = '../Progetti esercizi/6 Language/'
IMAGE_DIR = BASE_DIR + 'images'
IMAGE_SIZE = 224

label = {'Hello': 0, 'Yes': 1, 'No': 2, 'Thank you': 3, 'I love you': 4, 'Please': 5}

def load_data(path):
    images = []
    labels = []

    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            label_file = os.path.join(path, filename[:-4] + '.xml')
            tree = ET.parse(label_file)
            root = tree.getroot()
            label_text = root.find('object').find('name').text
            labels.append(label[label_text])

    return images, labels

images, labels = load_data(IMAGE_DIR)
images = np.array(images)
labels = np.array(labels)
labels = to_categorical(labels)

#using image augmentation and generate more data
datagen = ImageDataGenerator(
    rotation_range=2,
    shear_range=0.2,
    zoom_range=0.1,
    fill_mode='nearest')

TRAIN_AUG_DIR = BASE_DIR +'train'
TEST_AUG_DIR = BASE_DIR +'test'

train_gen = datagen.flow(images, labels, batch_size=32, save_to_dir=TRAIN_AUG_DIR, save_prefix='train', save_format='png')
test_gen = datagen.flow(images, labels, batch_size=32, save_to_dir=TEST_AUG_DIR, save_prefix='test', save_format='png')

#define model
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
vgg.trainable = False
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='softmax'))
#model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(train_gen, steps_per_epoch=len(images) / 32, epochs=8, validation_data=test_gen, validation_steps=len(images) / 32)
model.save('7lang')

#evaluate model
scores = model.evaluate(images, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

