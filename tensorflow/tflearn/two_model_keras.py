# -*- coding: utf-8 -*-
# @Time    : 10/28/16 16:49
# @Author  : Luke
# @Software: PyCharm
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

X = np.random.random((100,1))
y = np.random.random((100,1))

model = Sequential()
model.add(Dense(1,input_shape=([1])))
model.compile('sgd','mse')
model.fit(X,y)


model2 = Sequential()
model2.add(Dense(1,input_shape=([1])))
model2.compile('sgd','mse')
model2.fit(X,y)

print model2.predict([0.2])
