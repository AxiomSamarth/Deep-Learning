
# coding: utf-8

# In[1]:


import numpy as np
import nltk
from nltk.tokenize import word_tokenize


# In[2]:


from nltk.stem import WordNetLemmatizer
import random
import pickle
from collections import Counter


# In[3]:


lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


# In[4]:


def create_lexicon(pos, neg):
    lexicon = []
    
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    lexicon_2 = []
    
    for word in w_counts:
        if 1000 > w_counts[word] > 50:
            lexicon_2.append(word)
    
    return lexicon_2            


# In[5]:


def sample_handling(sample, lexicon, classification):
    featureset = []
    
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l)
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] = 1
            
            features = list(features)
            featureset.append([features, classification])
    
    return featureset            


# In[6]:


def create_feature_sets_and_labels(pos,neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)
    
    features = np.array(features)
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y


# In[8]:


if __name__ == 'main':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


# In[ ]:




