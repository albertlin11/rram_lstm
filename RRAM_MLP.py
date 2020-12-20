# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:46:33 2020

@author: HP ED-800
"""
import random as python_random
python_random.seed(1)
import numpy as np
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
import os

plt.close("all")
python_random.seed(1)
seed(1)
tf.random.set_seed(1)
os.environ['PYTHONHASHSEED'] = '0'

#a=pd.read_csv('ds_rram_MLP_HRS.csv')
a=pd.read_csv('./CSV_data/ds_rram_MLP_LRS.csv')

b=a.values

x_train=b[:,0].reshape((-1,1))
y_train=b[:,1].reshape((-1,1))
#bb=np.asarray(b, dtype='S')
#c=tf.convert_to_tensor(bb[:,2:], dtype=tf.string)
#tf.print(c)
#d=tf.data.Dataset.from_tensor_slices(c)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

x_train=scaler1.fit_transform(x_train)
#x_train=x_train.reshape((-1,5,2))
y_train=scaler2.fit_transform(y_train)

x = tf.keras.Input(shape=(1,))
x1=tf.keras.layers.Dense(units = 100, activation='relu')(x)
x2=tf.keras.layers.Dense(units = 100, activation='relu')(x1)
y=tf.keras.layers.Dense(units =  1)(x2)


regressor = tf.keras.Model(inputs=x, outputs=y, name='a_model')
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
log_dir= os.path.join('logs','fit','')
#log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


hist =regressor.fit(x_train, y_train, batch_size = 500, 
                    epochs = 2000,callbacks=[tensorboard_callback])
# save the training set
y_predict=regressor.predict(x_train)

################################# generate plot

x_train=scaler1.inverse_transform(x_train)
y_train=scaler2.inverse_transform(y_train)
y_predict=scaler2.inverse_transform(y_predict)


# data index as x
plt.figure(1)
plt.plot(y_train,'b',y_predict,'r')

# rram IV
plt.figure(2)
plt.plot(x_train,y_train,'b')
plt.plot(x_train,y_predict,'r')

# regression

plt.figure(3)
plt.plot([-1e-3,1e-3],[-1e-3,1e-3],'b')
plt.plot(y_train,y_predict,'+')


################################# saving files

df = pd.DataFrame (np.concatenate( (x_train, y_train, y_predict)  ,axis=1 ) )
filepath = './PaperData/MLP/Train_LRS.csv'
df.to_csv(filepath, header=['V_train','I_train','I_train_predict'] ,index=False)