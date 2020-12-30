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
def choose_data(pd_dataframe, data):
    return pd_dataframe.loc[:,data].to_numpy()

# feature scaling
def feat_scale(feature):
    return (feature - feature.mean()) / (feature.std())


# compose the desired matrix
def make_X(*x_data):
    m = len(x_data[0])
    Xmat = np.ones(m)
    for x in x_data:
        try:
            Xmat = np.column_stack((Xmat, x))
        except ValueError:
            raise ValueError('Input arrays not in the length, try again')
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
    
    # initilize parameters
    temp_theta = np.zeros(k)
    for i in range(0, len(theta)):
        temp_theta[i] = theta[i]
    theta = np.array(temp_theta)
    
    # initial cost
    J_diff = 10
    J = Jcost(theta, X, y)

    # batch gradient descending, default
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
mpg = choose_data(mtdata, 'mpg')
hp = choose_data(mtdata, 'hp')
wt = choose_data(mtdata, 'wt')    
drat = choose_data(mtdata, 'drat')
gear = choose_data(mtdata, 'gear')
qsec = choose_data(mtdata, 'qsec')
carb = choose_data(mtdata, 'carb')

mpg_nor = feat_scale(mpg)
wt_nor = feat_scale(wt)
hp_nor = feat_scale(hp)
drat_nor = feat_scale(drat)
gear_nor = feat_scale(gear)
qsec_nor = feat_scale(qsec)
carb_nor = feat_scale(carb)


theta, h = gradient_desc(mpg_nor, wt_nor, hp_nor, drat_nor, gear_nor, 
                         qsec_nor, carb_nor,alpha = 0.01)
#
#  analytical results

X = make_X(wt_nor, hp_nor, drat_nor, gear_nor, qsec_nor, carb_nor)
y = mpg_nor

theta_analytical = np.matmul(np.matmul(lin.inv(np.matmul(X.T, X)), X.T),y)

