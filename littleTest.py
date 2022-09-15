from gpConstr import *
import numpy as np

myGP = ConstrainedGaussianProcess(4)

myGP.addSample(np.array([3.0, 2.5, 1.0, 1.0]), 3.2)
myGP.addSample(np.array([5.0, 2.5, 1.0, 1.3]), 2.5)
myGP.addSample(np.array([1.4, 1.5, -4.2, 1.0]), -2.2)

myGP.trainWithoutConstraints()
# print(myGP.data_x)
# print(myGP.data_y)

# print(myGP.posteriorMeanWithoutConstraints(np.array([1.4, 1.5])))
# print(myGP.posteriorVarianceWithoutConstraints(np.array([1.4, 1.5])))

myGP.addDerivativeSample(np.array([3.0, 2.5, 1.0, 1.0]), 3)
myGP.addDerivativeSample(np.array([5.0, 2.5, 1.0, 1.3]), 2.5)
myGP.addDerivativeSample(np.array([1.4, 1.5, -4.2, 1.0]), -2.2)

print(myGP.data_der_x)
print(myGP.data_der_y)

myGP.trainConstraints()

print( myGP.posteriorMeanConstraints( np.array([3.0, 2.5, 1.0, 1.0]) ) ) 
print( myGP.posteriorMeanConstraints( np.array([3.1, 2.4, 1.0, 1.1]) ) ) 