# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 19:31:40 2021

@author: jacob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


# prepare for the data
am, drat, wt = choose_data(mtdata, 'am', 'drat', 'wt')
drat, wt = feat_scale(drat, wt)
newdata = pd.DataFrame(dict(drat = drat, wt = wt, am = am))

theta, h = gradient_desc_logistic(am, drat, wt, convergence=1e-6)

a = -(theta[0]/theta[2])
b = -(theta[1]/theta[2])

boundary = a + b*drat

fig1, ax = plt.subplots(dpi=500)
sns.scatterplot('drat','wt',data=newdata,hue='am',ax=ax)
sns.lineplot(drat, boundary, ax=ax, color = 'r')
plt.show()

# another example
am, wt, qsec = choose_data(mtdata, 'am', 'wt', 'qsec')
wt, qsec = feat_scale(wt, qsec)
newdata = pd.DataFrame(dict(wt = wt, qsec = qsec, am = am))

theta, h = gradient_desc_logistic(am, qsec, wt)

a = -(theta[0]/theta[2])
b = -(theta[1]/theta[2])

boundary = a + b*qsec

fig2, ax = plt.subplots(dpi=500)
sns.scatterplot('qsec','wt',data=newdata,hue='am',ax=ax)
sns.lineplot(qsec, boundary, ax=ax, color = 'r')
plt.show()