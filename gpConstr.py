import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import expm

class gpParameters():
    def __init__(self, order_input):
        self.order_input = order_input          # Input dimension
        self.sigma_err = 0.1                    # Measurement noise variance
        self.sigma_der_err = 0.8
        self.L = np.eye(self.order_input)       # Length scales matrix
        self.total_samples = 0                  # Total samples
        self.total_der_samples = 0              # Total derivative samples

class ConstrainedGaussianProcess():
    def __init__(self, order_input):
        self.params = gpParameters(order_input)
    
    #def dynamics(self, x):
    #    # Dynamic of the system (to be changed depending on the application)
    #    return np.array([x[2], x[3], 0, 0]).reshape((self.params.order_input, 1))

    def dynamics(self, x):
        # Dynamic of the system (to be changed depending on the application)
        return np.array([1])

    def k(self, x1, x2):
        # Reshape vectors to avoid computation problems
        x1 = x1.reshape((self.params.order_input, 1))
        x2 = x2.reshape((self.params.order_input, 1))

        # Compute kernel
        return math.exp( - (x1-x2).T @ np.linalg.inv(self.params.L) @ (x1 - x2) / 2)
    
    def dkdx(self, x1, x2):
        # Reshape vectors to avoid computation problems
        x1 = x1.reshape((self.params.order_input, 1))
        x2 = x2.reshape((self.params.order_input, 1))

        return -(x1 - x2).T @ np.linalg.inv(self.params.L) * self.k(x1, x2)
    
    def dkdx1dx2(self, x1, x2):
        # Reshape vectors to avoid computation problems
        x1 = x1.reshape((self.params.order_input, 1))
        x2 = x2.reshape((self.params.order_input, 1))

        return np.linalg.inv(self.params.L).T * self.k(x1, x2) + \
                np.linalg.inv(self.params.L).T @ (x1 - x2) @ (x1 - x2).T @ np.linalg.inv(self.params.L) * self.k(x1, x2)

    def addSample(self, x, y):
        # Add sample to the dataset
        if self.params.total_samples == 0:
            self.data_x = x.reshape((self.params.order_input, 1))
            self.data_y = np.array([y]).reshape((1, 1))
        else:
            self.data_x = np.append(self.data_x, x.reshape((self.params.order_input, 1)), 1)
            self.data_y = np.append(self.data_y, np.array(y).reshape((1,1)), 0)
        
        self.params.total_samples = self.params.total_samples + 1
    
    def addDerivativeSample(self, x, y):
        # Add sample to the dataset
        if self.params.total_der_samples == 0:
            self.data_der_x = x.reshape((self.params.order_input, 1))
            self.data_der_y = np.array([y]).reshape((1, 1))
        else:
            self.data_der_x = np.append(self.data_der_x, x.reshape((self.params.order_input, 1)), 1)
            self.data_der_y = np.append(self.data_der_y, np.array(y).reshape((1,1)), 0)
        
        self.params.total_der_samples = self.params.total_der_samples + 1

    def trainWithoutConstraints(self):
        # Train the gaussian process
        self.K_wC = np.zeros((self.params.total_samples, self.params.total_samples))
        for row in range(self.params.total_samples):
            for col in range(self.params.total_samples):
                self.K_wC[row, col] = self.k(self.data_x[:, row], self.data_x[:, col])
        self.K_wC = self.K_wC + self.params.sigma_err * np.eye(self.params.total_samples)
        self.L_chol_wC = np.linalg.cholesky(self.K_wC)
        self.alpha_wC = np.linalg.solve(self.L_chol_wC.T, np.linalg.solve(self.L_chol_wC, self.data_y))

    def posteriorMeanWithoutConstraints(self, x):
        k = np.zeros((self.params.total_samples, 1))
        for i in range(self.params.total_samples):
            k[i] = self.k(self.data_x[:, i], x)

        return k.T @ self.alpha_wC
    
    def posteriorVarianceWithoutConstraints(self, x):
        k = np.zeros((self.params.total_samples, 1))
        for i in range(self.params.total_samples):
            k[i] = self.k(self.data_x[:, i], x)

        return self.k(x, x) - k.T @ self.K_wC @ k
    
    def trainConstraints(self):
        self.K_11 = np.zeros((self.params.total_samples, self.params.total_samples))
        self.K_12 = np.zeros((self.params.total_samples, self.params.total_der_samples))
        self.K_21 = np.zeros((self.params.total_der_samples, self.params.total_samples))
        self.K_22 = np.zeros((self.params.total_der_samples, self.params.total_der_samples))
        
        # Fill K_11
        for row in range(self.params.total_samples):
            for col in range(self.params.total_samples):
                self.K_11[row, col] = self.k(self.data_x[:, row], self.data_x[:, col])
        
        # Fill K_12
        for row in range(self.params.total_samples):
            for col in range(self.params.total_der_samples):
                self.K_12[row, col] = self.dkdx(self.data_x[:, row], self.data_der_x[:, col]) @ self.dynamics(self.data_der_x[:, col])
        
        # Fill K_21
        for row in range(self.params.total_der_samples):
            for col in range(self.params.total_samples):
                self.K_21[row, col] = self.dkdx(self.data_der_x[:, row], self.data_x[:, col]) @ self.dynamics(self.data_der_x[:, row])
    
        # Fill K_22
        for row in range(self.params.total_der_samples):
            for col in range(self.params.total_der_samples):
                self.K_22[row, col] = self.dynamics(self.data_der_x[:, col]).T @ self.dkdx1dx2(self.data_der_x[:, row], self.data_der_x[:, col]) @ self.dynamics(self.data_der_x[:, row])
        
        # Construct block matrices and compute alpha
        self.K_C = np.block([ 
            [self.K_11, self.K_12],
            [self.K_21, self.K_22]
        ])

        err_I = np.block([ 
            [self.params.sigma_err * np.eye(self.params.total_samples), np.zeros((self.params.total_samples, self.params.total_der_samples))],
            [np.zeros((self.params.total_der_samples, self.params.total_samples)), self.params.sigma_der_err * np.eye(self.params.total_der_samples)]
        ])
        self.K_C = self.K_C + err_I

        self.L_chol_C = np.linalg.cholesky(self.K_C)
        self.alpha_C = np.linalg.solve(self.L_chol_C.T, np.linalg.solve(self.L_chol_C, np.block( [[self.data_y], [self.data_der_y] ]) ) )
    
    def posteriorMeanConstraints(self, x):
        k_1 = np.zeros((self.params.total_samples, 1))
        for i in range(self.params.total_samples):
            k_1[i] = self.k(self.data_x[:, i], x)
        
        k_2 = np.zeros((self.params.total_der_samples, 1))
        for i in range(self.params.total_der_samples):
            k_2[i] = self.dkdx(self.data_der_x[:, i], x) @ self.dynamics(self.data_der_x[:, i])
        
        k_3 = np.zeros((self.params.total_samples, 1))
        for i in range(self.params.total_samples):
            k_3[i] = -self.dkdx(self.data_x[:, i], x) @ self.dynamics(x)
        
        k_4 = np.zeros((self.params.total_der_samples, 1))
        for i in range(self.params.total_der_samples):
            k_4[i] = self.dynamics(x).T @ self.dkdx1dx2(self.data_der_x[:, i], x) @ self.dynamics(self.data_der_x[:, i])
         

        k_u = np.block( [
            [k_1],
            [k_2]
        ])

        k_f = np.block( [
            [k_3],
            [k_4]
        ])

        return k_u.T @ self.alpha_C, k_f.T @ self.alpha_C 

