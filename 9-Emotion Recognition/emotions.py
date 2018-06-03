
# coding: utf-8

from __future__ import division
import pandas as pd
import cv2
import numpy as np
from random import shuffle


import os
import tflearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 48
LR = 1e-3
MODEL_NAME = 'emotion-{}-{}.model'.format(LR, '12conv-basic')


def create_training_data(emotion,pixels):
    
    training_data = []
    for index, data in enumerate(pixels):
        data = data.split(' ')
        data = np.array(data, np.float64)
        label = np.zeros(7)
        label[emotion[index]] = 1        
        training_data.append([data, label])
        
    shuffle(training_data)
    #print(len(training_data))
    np.save('training_data.npy',training_data)
    return training_data

#df = pd.read_csv('Dataset/dataset.csv')
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
#emotion = df['emotion'][:15000]
#pixels = df['pixels'][:15000]

#training_data = create_training_data(emotion,pixels)
training_data = np.load('training_data.npy')


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 7, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')


train = training_data[:-500]
test = training_data[-500:]

X = np.array([i[0] for i in train], dtype=np.float64).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train], dtype=np.float64)

test_x = np.array([i[0] for i in test], dtype=np.float64).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y =np.array([i[1] for i in test], dtype=np.float64)

#model.fit({'input': X}, {'targets': Y}, n_epoch=75, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#model.save(MODEL_NAME)

#model_out.index(max(model_out))
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral


def find_emotion(image):
    image = image.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([image])[0]
    result = model_out.index(max(model_out))
    emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

    return emotion[result]




