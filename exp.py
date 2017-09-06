# -*- coding: utf-8 -*-
# @Time    : 10/13/16 09:51
# @Author  : Luke
# @Software: PyCharm
from scipy import stats
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt

a =np.random.randn(10000)*20+100
b =np.random.randn(10000)*10+50

x1 = np.histogram(a,100)
x2 = np.histogram(b,100)



plt.plot(x1[1][:-1],x1[0])
plt.plot(x2[1][:-1],x2[0])
# plt.plot(b)
# plt.show()

gmm = mixture.GaussianMixture(n_components=2)
gmm.fit(np.stack([a,b]).reshape(-1,1))

gmm.predict(50)


x = np.arange(0,200,1)
y1 = stats.norm.pdf(x,gmm.means_[0],np.sqrt(gmm.covariances_[0])).reshape(x.shape)*50*a.max()
y2 = stats.norm.pdf(x,gmm.means_[1],np.sqrt(gmm.covariances_[1])).reshape(x.shape)*50*a.max()

plt.plot(x,y1)
plt.plot(x,y2)
plt.show()