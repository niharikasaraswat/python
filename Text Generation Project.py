#!/usr/bin/env python
# coding: utf-8

# In[5]:


import keras


# In[32]:


pip install sequential


# In[29]:


pip install keras


# In[7]:


import nltk


# In[2]:


import sequential


# In[5]:


# import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[26]:


pip install nltk


# In[7]:


# load data
file = open ("frankenstein.txt").read()


# In[16]:


pip install tensorflow


# In[8]:


# tokenization
# standardization
def tokenize_words(input):
    input = input.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'),tokens)
    return "".join(filtered)

processed_inputs = tokenize_words(file)


# In[9]:


# chars to numbers
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i,c in enumerate(chars))


# In[10]:


# check if words to chars or chars to num(?!) has worked
input_len = len(processed_inputs)
vocab_len = len(chars)
print("total number of characters: ",input_len)
print("total vocab :",vocab_len)


# In[11]:


#seq_length
seq_length = 100
x_data = []
y_data = []


# In[12]:


#loop through sequence
for i in  range(0,input_len - seq_length,1):
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char]for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print("Total patterns : ",n_patterns)


# In[13]:


# convert input sequence to np array and so on
X = numpy.reshape(x_data, (n_patterns,seq_length,1))
X = X/float(vocab_len)


# In[14]:


# one-hot coding
y = np_utils.to_categorical(y_data)


# In[15]:


# creating the model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(256 , return_sequences=True))
model.add(Dropout(0,2))
model.add(LSTM(128))
model.add(Dropout(0,2))
model.add(Dense(y.shape[1],activation='softmax'))


# 

# In[16]:


#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam')


# In[17]:


# saving weights
filepath= 'model_weights_saved.hdf5'
checkpoint = ModelCheckpoint(filepath,monitor='loss' , verbose = 1,save_best_only=True,mode='min')
desired_callbacks = [checkpoint]


# In[24]:


# fit model and let it train
model.fit(X,y,epochs= 4, batch_size=256 , callbacks = desired_callbacks)


# In[25]:


# recompile model with saved weights
filename = 'model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')


# In[26]:


# output of model back into characters
num_to_char = dict((i,c)for i,c in enumerate(chars))


# In[36]:


#random seed to hrelp generate
start = numpy. random.randint(0, len(x_data)- 1)
pattern = x_data[start]
print("Random seed: ")
print("\" ", ''.join([num_to_char[value]for value in pattern]),"\"")


# In[37]:


#generate the text
for i in range(1000):
    x = numpy.reshape(pattern, (1,len(pattern), 1))
    x = x/float(vocab_len)
    prediction = model.predict(x,verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]


# In[ ]:




