#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt

dp=np.load('data_parallel.npy')
mp=np.load('model_parallel.npy')
single=np.load('single_cpu.npy')

#print dp[:10]
#print single[:10]

plt.plot(dp,"r")
plt.hold(True)
plt.plot(single,"b")
#plt.plot(mp,"g")
plt.yscale('log')
plt.grid()
plt.xlabel("loop")
plt.ylabel("loss")
plt.savefig("compare.png")


