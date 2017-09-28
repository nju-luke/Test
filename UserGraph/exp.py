from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
# 
def func(x, a, b):
    return a * np.exp(-b * x)

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
plt.plot(xdata,ydata,'b-')
popt, pcov = curve_fit(func, xdata, ydata)
y2 = [func(i, popt[0],popt[1]) for i in xdata]
plt.plot(xdata,y2,'r--')
plt.show()
print popt



# x = np.arange(200)*0.01
# y1 = np.exp(-x)
# y2 = 2*(np.exp(x) - 1)
#
# plt.plot(x,y1)
# plt.plot(x,y2)
#
# plt.show()





