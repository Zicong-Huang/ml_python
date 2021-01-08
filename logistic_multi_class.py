# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:22:52 2021

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
    


# see correlation
cyl, mpg, wt = choose_data(mtdata, 'cyl', 'mpg', 'wt')
mpg, wt = feat_scale(mpg, wt)

fig1, ax = plt.subplots(dpi=300)
sns.violinplot(x=cyl, y=mpg, ax=ax)
ax.set_xlabel('cyl')
ax.set_ylabel('mpg')

fig2, ax = plt.subplots(dpi=300)
sns.violinplot(x=cyl, y=wt, ax=ax)
ax.set_xlabel('cyl')
ax.set_ylabel('wt')

newdata = pd.DataFrame(dict(cyl = cyl, wt = wt, mpg = mpg))

fig3, ax = plt.subplots(dpi=300)
sns.scatterplot(newdata['wt'], newdata['mpg'], hue=newdata['cyl'], ax=ax, legend='full',
                palette=['green','red','blue'])

# one vs all classification
cyl_4 = np.array((cyl == 4)) * 1
cyl_8 = np.array((cyl == 8)) * 1

# boundary for 4, 6, 8
theta_4, h_4 = gradient_desc_logistic(cyl_4, mpg, wt)
a_4 = -(theta_4[0]/theta_4[2])
b_4 = -(theta_4[1]/theta_4[2])
boundary_4 = a_4 + b_4*mpg

theta_8, h_8 = gradient_desc_logistic(cyl_8, mpg, wt)
a_8 = -(theta_8[0]/theta_8[2])
b_8 = -(theta_8[1]/theta_8[2])
boundary_8 = a_8 + b_8*mpg

fig4, ax = plt.subplots(dpi=300)
sns.scatterplot(newdata['mpg'], newdata['wt'], hue=newdata['cyl'], ax=ax, legend='full',
                palette=['green','red','blue'])
sns.lineplot(x=mpg, y=boundary_4)
sns.lineplot(x=mpg, y=boundary_8)