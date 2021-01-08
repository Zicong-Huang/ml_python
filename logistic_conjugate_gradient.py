# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:31:10 2021

@author: jacob
"""

import numpy as np
import pandas as pd
from scipy import optimize as op
from scipy.special import expit
# read data
mtdata = pd.read_csv('../ml_python/sample_data/mtcars.csv')

# prepare variables
def choose_data(pd_dataframe, *data):
    chosen = list()
    for d in data:
        chosen.append(pd_dataframe.loc[:, d].to_numpy())
    return chosen

# feature scaling
def feat_scale(*feature):
    outcome = list()
    for f in feature:
        f_out = (f - f.mean()) / f.ptp()
        outcome.append(f_out)
    return outcome
    
# compose the desired matrix
def make_X(*x_data):
    m = len(x_data[0])
    Xmat = np.ones(m)
    for x in x_data:
        Xmat = np.column_stack((Xmat, x))
    return Xmat

# cost function
def Jcost(theta, X, y):
    m = len(y)
    h = expit(X @ theta)
    cost = -(y*np.log(h))-((1-y)*np.log(1-h))
    J =  (1/m)*np.sum(cost)
    return J

# gradient
def Jcost_prime(theta, X, y):
    m = len(y)
    return (1/m) * (X.T @ (X @ theta - y))

# conjugated gradient descend
drat, am = choose_data(mtdata, 'drat', 'am')        # extract data
drat_nor, = feat_scale(drat)             # scaling
X = make_X(drat_nor)                                # construct the data matrix
m,k = X.shape
y = am                                          # construct the y vector
theta0 = np.array([0,10])                                # initial guess of tehta

res = op.fmin_cg(Jcost, theta0, fprime = Jcost_prime, args=(X, y))
