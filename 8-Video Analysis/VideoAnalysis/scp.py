
# coding: utf-8

# In[1]:


import numpy as np
import os
from tqdm import tqdm
from random import shuffle
from collections import Counter


# In[2]:


import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# In[3]:


IMG_SIZE = 100
LR = 1e-3
MODEL_NAME = 'video-{}-{}.model'.format(LR, '12conv-basic')


# <p>Now define the functions to label the video frames and to prepare the training dataset as well</p>

# In[4]:


def label_img(image):
    name = image.split('.')
    if name[0] == 'happy':
        return [1, 0]
    elif name[0] == 'fight':
        return [0, 1]


# In[5]:


def create_train_data():
    
    training_data = [] 
    for image in tqdm(os.listdir('./train/')):
        label = label_img(image)
        pixel = cv2.resize(cv2.imread('./train/'+image, 0),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(pixel), np.array(label)])
    
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data


# In[6]:


#training_data = create_train_data()


# In[7]:


#if the training data is already available then use the following
#training_data = np.load('training_data.npy')


# In[8]:


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

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[9]:


if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded')


# In[10]:

'''
train = training_data[:-500]
test = training_data[-500:]


# In[11]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


# In[12]:


#model.fit({'input': X}, {'targets': Y}, n_epoch=25, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#model.save(MODEL_NAME)


# In[13]:


total = len(os.listdir('./fight/avengers'))
count = 0
for image in tqdm(os.listdir('./fight/avengers')):
    #name = image.split('.')[0]
    data = cv2.imread('./fight/avengers/'+image, 0).reshape(IMG_SIZE, IMG_SIZE, 1)
    output = model.predict([data])[0].index(max(model.predict([data])[0]))
    if output == 1:
        count+=1
'''

# In[14]:


#print((count/total)*100)


# In[15]:


#data = cv2.resize(cv2.imread('./fight/THE AVENGERS.mp40002.bmp', 0), (IMG_SIZE, IMG_SIZE)).reshape(IMG_SIZE, IMG_SIZE, 1)
#print(model.predict([data])[0].index(max(model.predict([data])[0])))


# In[17]:

def predict(path):
    count = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        
        if frame == None:
            break
        else:
            #cv2.imshow('frame', frame)
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE))
            result = model.predict([frame.reshape(IMG_SIZE, IMG_SIZE, 1)])[0]
            count.append(result.index(max(result)))
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #print(len(count))
    context = Counter(count)
    #print(context.most_common())
    if context.most_common()[0][0] == 0:
        return('Happy moment!')
    else:
        return('Fight and rage')


# In[ ]:




