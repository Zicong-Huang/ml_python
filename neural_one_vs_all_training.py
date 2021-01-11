import numpy as np
import scipy.optimize as scop
from logistic_cost_function import logistic_cost_function


def one_vs_all_training(x, y, lamb, num_iter):
    m, n = x.shape
    num_label = 10

    all_theta = np.zeros((num_label, n+1))
    x = np.insert(x, 0, np.ones(m), axis=1)

    initial_theta = np.zeros((n+1, 1))
    for i in range(num_label):
        print('processing on:', i, end='\r')
        y_is_i = (y == i)
        theta = scop.minimize(logistic_cost_function, initial_theta, args=(x, y_is_i, lamb), method='CG', options=dict(maxiter=num_iter))
        all_theta[i, :] = theta.x

    return all_theta
