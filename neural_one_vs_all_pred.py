from scipy.special import expit
import numpy as np


def one_vs_all_predict(trained_theta, x):
    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    z = x @ trained_theta.T
    a = expit(z)
    p = np.argmax(a, axis=1)
    return p
