from scipy.special import expit
import numpy as np


def logistic_cost_function(theta, x, y, lamb=0, return_gradient=False):
    m, k = x.shape
    h = expit(x @ theta)     # implement the sigmoid function
    j = -(1/m)*(np.sum(y*np.log(h)+(1-y)*np.log(1-h))) + (lamb/(2*m))*np.sum(theta[1:]**2)

    gradient = np.zeros(k)
    gradient[0] = (1/m) * (x.T[0, :] @ (h - y))
    gradient[1:] = (1/m) * (x.T @ (h-y))[1:] + (lamb/m)*theta[1:]

    if return_gradient:
        return j, gradient
    else:
        return j
