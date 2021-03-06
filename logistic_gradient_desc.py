# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:26:50 2020

@author: jacob
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    print(np.min(linear))
    h = logistic(linear)
    return h

# cost function
def Jcost(theta, X, y):
    m = len(y)
    h = hypo(theta, X)
    cost = -(y*np.log(h))-((1-y)*np.log(1-h))
    J =  (1/m)*np.sum(cost)
    return J


# update theta
def gradient_update(X, y, theta, alpha, m):
    h = hypo(theta, X)
    new_theta = theta-(alpha/m)*(X.T @ (h-y))
    return new_theta


# gradient descending: set up
def gradient_desc_logistic(y_data, *x_data, alpha=0.01, 
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
    J = Jcost(theta, X, y)
    
    # batch gradient descending
    Js = list()
    thetas = list()
    while J_diff > convergence:
        Js.append(J)
        thetas.append(theta)

        update_theta = gradient_update(X, y, theta, alpha, m)
        update_J = Jcost(update_theta, X, y)
        
        J_diff = J - update_J
        if J_diff < 0:
            raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
        
        J = update_J.copy()
        theta = update_theta

    # calculate hypothesis h
    h = hypo(theta, X)
    
    yield theta
    yield h
    if rec_theta == True: yield thetas
    if rec_J == True: yield Js



# implement gradient descending
am, drat, wt, qsec = choose_data(mtdata, 'am', 'drat', 'wt', 'qsec')
drat_nor, wt_nor, qsec_nor = feat_scale(drat, wt, qsec)
theta, h, Js = gradient_desc_logistic(am, drat_nor, wt_nor, qsec_nor, alpha=0.1,
                                  convergence=1e-6, rec_J=True)


# some plots
fig1, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=drat)
ax.set_ylabel('drat')
ax.set_xlabel('am')
ax.set_title('am distribution against drat')
plt.show()

fig2, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=wt)
ax.set_ylabel('wt')
ax.set_xlabel('am')
ax.set_title('am distribution against wt')
plt.show()

fig3, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=qsec)
ax.set_ylabel('qsec')
ax.set_xlabel('am')
ax.set_title('am distribution against qsec')
plt.show()

fig4, ax = plt.subplots(dpi=200)
ax.plot(Js)
plt.show()