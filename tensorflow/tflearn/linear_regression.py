""" Linear Regression Example """

from __future__ import absolute_import, division, print_function

import tflearn
import tensorflow as tf

lr = 0.01

# Regression data
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)

regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=lr)
m = tflearn.DNN(regression,tensorboard_dir="./tflearn_logs/",tensorboard_verbose=3)
m.fit(X, Y, n_epoch=1, show_metric=True, snapshot_epoch=False)

model = tflearn.DNN(linear,session=m.session)
y = model.predict([3])
model1 = model
print(y)