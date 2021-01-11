# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 01:37:41 2021

@author: jacob
"""

import numpy as np
import pandas as pd

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


# logistic function
def logistic(z):
    return 1/(1+np.exp(-z))

# hypothesis function
def hypo(theta, X):
    linear = X @ theta
    h = logistic(linear)
    return h

# extract theta_1 - theta_n for regularization use
def theta_no_cons_term(theta):
    new_theta = np.insert(theta[1:], 0, 0)
    return new_theta

# cost function
def Jcost(theta, X, y, lamb):
    m = len(y)
    h = hypo(theta, X)
    cost = -(y*np.log(h))-((1-y)*np.log(1-h))
    new_theta = theta_no_cons_term(theta)
    J =  (1/m)*np.sum(cost) + (lamb/(2*m))*np.sum((new_theta**2))
    return J


# update theta
def gradient_update(X, y, theta, alpha, m, lamb):
    h = hypo(theta, X)
    # update theta_0
    theta[0] = theta[0] - (alpha/m)*(X.T @ (h-y))[0]
    # update theta_1 - theta_n
    theta[1:] = theta[1:]*(1-(alpha*lamb)/m) - (alpha/m)*(X.T @ (h-y))[1:]
    return theta

# gradient descending: set up
def gradient_desc_logistic(y_data, *x_data, alpha=0.01, lamb=0, 
                           convergence=1e-6, theta=[], 
                           rec_theta=False, rec_J=False):
    # set up the data
    X = make_X(*x_data)
    y = y_data
    m,k = X.shape
    
    # initialize parameters
    temp_theta = np.zeros(k)
    for i in range(0, len(theta)):
        temp_theta[i] = theta[i]
    theta = np.array(temp_theta)
    
    # initial cost
    J_diff = 10
    J = Jcost(theta, X, y, lamb)
    
    # batch gradient descending
    Js = list()
    thetas = list()
    while J_diff > convergence:
        Js.append(J)
        thetas.append(theta)

        update_theta = gradient_update(X, y, theta, alpha, m, lamb)
        update_J = Jcost(update_theta, X, y, lamb)
        
        J_diff = J - update_J
        if J_diff < 0:
            raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
        
        J = update_J
        theta = update_theta
        print(J)

    # calculate hypothesis h
    h = hypo(theta, X)
    
    yield theta
    yield h
    if rec_theta == True: yield thetas
    if rec_J == True: yield Js
    



