
# coding: utf-8

# In[14]:

from __future__ import division
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
import pickle
from create_sentiment_featureset import create_feature_sets_and_labels, create_lexicon
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[2]:


lemmatizer = WordNetLemmatizer()


# In[3]:


LR = 1e-3
MODEL_NAME = 'sentiment-{}-{}.model'.format(LR, '6conv-basic')


# In[4]:


train_x, train_y, test_x, test_y = pickle.load(open('sentiment_set.pickle', 'rb'))


# In[5]:


train_x = np.array(train_x).reshape(-1, 1, 423, 1)
train_y = train_y
test_x = np.array(test_x).reshape(-1, 1, 423, 1)
test_y = test_y


# In[6]:


convnet = input_data(shape=[None, 1, 423, 1], name='input')

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


# In[7]:


if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')


# In[8]:


X = train_x
Y = train_y
print(X.shape)


# In[9]:


#model.fit({'input': X}, {'targets': Y}, n_epoch=25, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
#model.save(MODEL_NAME)


# In[10]:


def process_prediction_sample(sentence):
    words = word_tokenize(sentence)
    words = [lemmatizer.lemmatize(i) for i in words]
    
    lexicon = pickle.load(open('lexicon.pickle','rb'))
    #print(type(lexicon))
    
    features = np.zeros(len(lexicon))
    for word in words:
        if word.lower() in lexicon:
            index_value = lexicon.index(word.lower())
            features[index_value] = 1
    
    return features


# In[11]:


# uncomment the commented section to predict the sentiment for a single input statement
# input_data = "Sentiment Analyzer is working fine."
# features = process_prediction_sample(input_data)
# features = np.array(features).reshape(-1, 1,423,1)


# In[12]:


# model_out = model.predict(features)[0]
# print (model_out)
# if np.argmax(model_out) == 0:
#    str_label = 'Positive'
# else:
#    str_label = 'Negative'


# In[13]:


#print(str_label)


# In[20]:


def custom_accuracy_check():
    test_data = []
    for file in ['pos.txt', 'neg.txt']:
        with open(file, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                test_data.append([line, file[:3]])
    
    shuffle(test_data)
    test_data = test_data[:5000]
    correct = 0
    total = len(test_data)
    for data in test_data:
        features = process_prediction_sample(data[0])
        label = data[1]
        features = np.array(features).reshape(-1, 1,423,1)
        
        model_out = model.predict(features)[0]
        if np.argmax(model_out) == 0:
            str_label = 'pos'
        else:
            str_label = 'neg'
        
        if label == str_label:
            correct += 1
    
    print ("The percentage accuracy is",(correct/total)*100)        
        


# In[21]:


custom_accuracy_check()


# In[ ]:




