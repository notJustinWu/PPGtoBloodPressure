import pandas as pd 
import numpy as np  
import torch.nn as nn
import pandas as pd    
import sys
import os
import torch
import csv
import scipy.io as sio
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as Data          
import matplotlib.pyplot as plt        
import tensorflow as tf
from keras.layers import BatchNormalization
from torch.autograd import Variable
from tensorflow import keras
from tensorflow import keras 
from keras.models import Model
from torch import nn
#from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Dropout, Bidirectional, Conv1D

import keras.api._v2.keras as keras
from keras.layers import Input, LSTM, Dense, Bidirectional, Conv1D, ReLU, TimeDistributed
'''
ppg = pd.read_csv('preprocessed/ppg_pp.csv')
abp = pd.read_csv('preprocessed/abp_pp.csv')
ecg = pd.read_csv('preprocessed/ecg_pp.csv')
dbp = pd.read_csv('preprocessed/dbp_pp.csv')
sbp = pd.read_csv('preprocessed/sbp_pp.csv')

ppg = pd.read_csv('data/ppg.csv')
abp = pd.read_csv('data/abp.csv')
ecg = pd.read_csv('data/ecg.csv')
'''
ppg = pd.read_csv('preprocessed/ppg_pp.csv')
abp = pd.read_csv('preprocessed/abp_pp.csv')
ecg = pd.read_csv('preprocessed/ecg_pp.csv')
dbp = pd.read_csv('preprocessed/dbp_pp.csv')
sbp = pd.read_csv('preprocessed/sbp_pp.csv')
abp = np.divide(np.subtract(abp, 50), 150)
#data = np.stack((ppg, ecg), axis=-1)
#ppg = np.multiply(np.add(ppg,20), 5)
#ecg = np.multiply(np.add(ecg,5),25)
#data = np.stack((ppg), axis=-1)


#adjust train/test/valid data
X_train = ppg[:3500]
X_val = ppg[3500:4500]
X_test = ppg[4500:]

y_train = dbp[:3500]
y_val = dbp[3500:4500]
y_test = dbp[4500:]

for i in y_train:
    for j in i:
        print(j)
#plt.plot(y_train.iloc[10])
#plt.plot(ecg.iloc[10])
#plt.plot(ppg.iloc[10])
plt.show()
inputs = Input(shape=(1024, 1))

#conv = Conv1D(filters=32, kernel_size=5, strides=1,  activation='relu')(inputs)
# First LSTM cell
encoder = LSTM(512, return_sequences=True)(inputs)

# Second LSTM cell
decoder = LSTM(512, return_sequences=True)(encoder)
#sequence_prediction = Bidirectional(LSTM(64, return_sequences=True)(decoder))

# Dense layer applied to each time step of the second LSTM cell output
sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

# Define and compile the model
model = Model(inputs, sequence_prediction)
model.compile('adam', 'mae')

# Fit the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the test data
model.evaluate((X_test),y_test)

# Make predictions using the model
prediction = model.predict(X_test)


plt.plot(y_test.iloc[10])
plt.plot(prediction[10])

plt.savefig('graph.png')
plt.show()

'''
for i in range(3500,4500):
    for j in range(0,1000):
        sum += abs(((150*prediction[i][j])+50) - ((150*y_test[i][j])+50))
  
error = sum/1000000
'''

# display
print("Mean absolute error : " + str(error))
