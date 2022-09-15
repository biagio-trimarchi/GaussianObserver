from gpConstr import *
import numpy as np
import matplotlib.pyplot as plt

myGP = ConstrainedGaussianProcess(1)

myGP.addSample(np.array([0]), 5*np.sin(0))
myGP.addSample(np.array([1]), 5*np.sin(1))

myGP.addSample(np.array([2]), 5*np.sin(2))
myGP.addSample(np.array([3]), 5*np.sin(3))

myGP.trainWithoutConstraints()

x = np.linspace(-2, 6, 100)
y1 = 5*np.sin(x)

y2 = []
for xx in x:
    y2.append( myGP.posteriorMeanWithoutConstraints(xx) )
y2 = np.array(y2).reshape((100,))

fig = plt.figure()
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='gp')
plt.legend()
plt.show()


myGP.addDerivativeSample(np.array([0]), 5*np.cos(0))
myGP.addDerivativeSample(np.array([1]), 5*np.cos(1))
myGP.addDerivativeSample(np.array([2]), 5*np.cos(2))
myGP.addDerivativeSample(np.array([3]), 5*np.cos(3))

myGP.trainConstraints()

y2 = []
for xx in x:
    y2.append( myGP.posteriorMeanConstraints(xx)[0])
y2 = np.array(y2).reshape((100,))

fig = plt.figure()
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='gp')
plt.legend()
plt.show()
