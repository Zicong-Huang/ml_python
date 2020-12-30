# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:26:46 2020

@author: jacob
"""

import numpy as np
from numpy import linalg as lin
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
        f_out = (f - f.mean()) / f.std()
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
    h = np.matmul(X,theta)
    return (1/(2*m)) * np.sum((h - y)**2)


# update theta
def gradient_update(X, y, theta, alpha, m):
    h = np.matmul(X,theta)
    delta = np.matmul(X.T, h - y)
    theta = theta - (alpha/m)*delta
    return theta


# gradient descending: set up
def gradient_desc(y_data, *x_data, alpha=0.01, 
                  convergence=1e-16, theta=[]):
    # set up the data
    X = make_X(*x_data)
    y = y_data
    m,k = X.shape
    
    # initailize parameters
    temp_theta = np.zeros(k)
    for i in range(0, len(theta)):
        temp_theta[i] = theta[i]
    theta = np.array(temp_theta)
    
    # initial cost
    J_diff = 10
    J = Jcost(theta, X, y)

    # batch gradient descending
    while J_diff > convergence:
        theta = gradient_update(X, y, theta, alpha, m)
        update_J = Jcost(theta,X,y)
        J_diff = J - update_J
        if J_diff < 0:
            raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
        J = update_J

    # calculate hypothesis h
    h = np.matmul(X, theta)
    
    return theta, h

#
# implementation

mpg,hp,wt,drat,gear,qsec,carb = choose_data(mtdata, 'mpg','hp','wt','drat',
                                            'gear','qsec','carb')

mpg_nor, wt_nor, hp_nor, drat_nor = feat_scale(mpg, wt, hp, drat)

gear_nor, qsec_nor, carb_nor = feat_scale(gear, qsec, carb)


theta, h = gradient_desc(mpg_nor, wt_nor, hp_nor, drat_nor, gear_nor, 
                         qsec_nor, carb_nor,alpha = 0.01)
#
#  analytical results

X = make_X(wt_nor, hp_nor, drat_nor, gear_nor, qsec_nor, carb_nor)
y = mpg_nor

theta_analytical = np.matmul(np.matmul(lin.inv(np.matmul(X.T, X)), X.T),y)

