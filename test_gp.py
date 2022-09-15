import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcess as GP

# Initializer - Declare Size of x (1 Here) and of y (1 Here)
fitter = GP(1, 1)

# Sample From sin Function
x = np.arange(-1, 3, 0.1)
y = np.sin(x)

# Add Samples
for idx in range(0, x.shape[0]):
  fitter.add_sample(x[idx], y[idx])

# Fit Hyperparameters - Not Mandatory (Very High Computational Complexity)
fitter.optimize_hyperparameters()

# Train
fitter.train()

# Get Inference & Variance
dom = np.arange(-2, 4, 0.1)
yy = np.empty(dom.shape)
yv = np.empty(dom.shape)
for idx in range(dom.shape[0]):
  yy[idx] = fitter.posterior_mean(dom[idx])
  yv[idx] = fitter.posterior_variance(dom[idx])

# fig = plt.figure()
# plt.plot(dom, yy, label='Gaussian Process Inference')
# plt.plot(dom, yy + yv, ':', label='Upper Bound Variance')
# plt.plot(dom, yy - yv, ':', label='Lower Bound Variance')
# plt.plot(dom, np.sin(dom), label='Real Function')
# plt.plot(x, y, '.', label='Sampled Points')
# plt.grid()
# plt.legend()

# Get Derivative
yd = np.empty(dom.shape)
yds = np.empty(dom.shape)
for idx in range(dom.shape[0]):
  yd[idx] = fitter.posterior_dxmean(dom[idx])
  yds[idx] = fitter.posterior_dxvariance(dom[idx])/(2*np.sqrt(yv[idx]))

fig = plt.figure()
plt.plot(dom, yd, label='Gaussian Process Derivative')
plt.plot(dom, yd+yds, ':', label='Upper Bound')
plt.plot(dom, yd-yds, ':', label='Lower Bound')
plt.plot(dom, np.cos(dom), label='Real Function')
plt.grid()
plt.legend()

fig = plt.figure()
plt.plot(dom, yv, label='Gaussian Process Variance')
plt.plot(dom, yds, label='Gaussian Process Variance Derivative')
plt.plot(x, 0*x, '.', label='Sampled Points')
plt.grid()
plt.legend()

# Get Derivative
ydd = np.empty(dom.shape)
for idx in range(dom.shape[0]):
  ydd[idx] = fitter.posterior_ddxmean(dom[idx])

# fig = plt.figure()
# plt.plot(dom, ydd, label='Gaussian Process Second Derivative')
# plt.plot(dom, -np.sin(dom), label='Real Function')
# plt.grid()
# plt.legend()

# fig = plt.figure()
# plt.plot(dom, yy - np.sin(dom), label='Gaussian Process Inference Error')
# plt.plot(dom, ydd + np.sin(dom), label='Gaussian Process Second Derivative Error')
# plt.plot(dom, yd - np.cos(dom), label='Gaussian Process Derivative Error')
# plt.grid()
# plt.legend()

# plt.show()

