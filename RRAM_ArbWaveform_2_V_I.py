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

a=pd.read_csv('./CSV_data/ds_rram_test_rnseed1.csv')
#a=pd.read_csv('./CSV_data/ds_rram_sine0.csv')
b=a.values

x_train=b[:,0:b.shape[1]-1]
y_train=b[:,10]
#bb=np.asarray(b, dtype='S')
#c=tf.convert_to_tensor(bb[:,2:], dtype=tf.string)
#tf.print(c)
#d=tf.data.Dataset.from_tensor_slices(c)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

x_train=scaler1.fit_transform(x_train)
x_train=x_train.reshape((-1,5,2))
y_train=scaler2.fit_transform(y_train.reshape((-1,1)))

x = tf.keras.Input(shape=(None, 2))
x1=tf.keras.layers.LSTM(units = 100, return_sequences=True)(x)
x2=tf.keras.layers.LSTM(units = 100, return_sequences=False)(x1)
y=tf.keras.layers.Dense(units =  1)(x2)


regressor = tf.keras.Model(inputs=x, outputs=y)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
log_dir= os.path.join('logs','fit','')
#log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# sine batch_szie=100  , rd walk batch_size=1000
hist =regressor.fit(x_train, y_train, batch_size = 1000, 
                    epochs = 2000,callbacks=[tensorboard_callback])
# save the training set
y_predict=regressor.predict(x_train)

plt.plot(y_train,'b',y_predict,'r')



# rram IV
xx_v=scaler1.inverse_transform(x_train.reshape(-1,10))
xx_v=(xx_v[:,8]).reshape((-1,1))
yy_I=scaler2.inverse_transform(y_train)
yy_I_pred=scaler2.inverse_transform(y_predict)
plt.figure(2)
plt.plot(xx_v,yy_I,'s')
plt.plot(xx_v,yy_I_pred,'+')

# regression

plt.figure(3)
plt.plot([-1e-3,1e-3],[-1e-3,1e-3],'b')
plt.plot(yy_I,yy_I_pred,'+')

################################# saving files

df = pd.DataFrame (np.concatenate( (xx_v, yy_I, yy_I_pred)  ,axis=1 ) )
filepath = './PaperData/RandomWalk/Train.csv'
#filepath = './PaperData/Sine/Train.csv'
df.to_csv(filepath, header=['V_train','I_train','I_train_predict'] ,index=False)