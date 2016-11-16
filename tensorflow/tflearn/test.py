# -*- coding: utf-8 -*-
# @Time    : 10/16/16 11:12
# @Author  : Luke
# @Software: PyCharm

import numpy as np
from tflearn import DNN
from tflearn import fully_connected
from tflearn import input_data
from tflearn import regression
import tflearn
import tensorflow as tf
X = np.random.random((100,1))
y = np.random.random((100,1))

network = input_data(shape=(None,1))
network = fully_connected(network,1,activation='sigmoid')
network = regression(network,optimizer='sgd',loss='mean_square')

model = DNN(network)
model.fit(X,y)
print model.predict([[2.]])


X = np.random.random((100,1))
y = np.random.random((100,1))

network1 = input_data(shape=(None,1))
network2 = fully_connected(network1,1,activation='sigmoid')
network3 = regression(network2,optimizer='sgd',loss='mean_square')

model2 = DNN(network3)
model2.fit(X,y)
print model2.predict([[2.]])





