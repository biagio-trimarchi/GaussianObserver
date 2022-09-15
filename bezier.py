import numpy as np
from math import comb

def bernsteinPolynomial(n, k, t):
    return comb(n, k) * (t ** k) * ( (1 - t) ** (n - k) )

def bezier(P, t):
    p = np.zeros((P.shape[0], 1))
    for k in range(P.shape[1]):
        p = p + bernsteinPolynomial(P.shape[1]-1, k, t) * P[:, k].reshape((P.shape[0], 1))
    return p
