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
    h = logistic(linear)
    return h

# cost function
def Jcost(theta, X, y, epsilon=0):
    m = len(y)
    h = hypo(theta, X)
    cost = (y*np.log(h+epsilon))**2+((1-y)*np.log(1-h+epsilon))**2
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
theta, h, Js = gradient_desc_logistic(am, drat_nor, wt_nor, qsec_nor, alpha=0.01,
                                  convergence=1e-15, rec_J=True)


# some plots
fig1, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=drat)
ax.set_ylabel('drat')
ax.set_xlabel('am')
ax.set_title('am distribution against drat')

fig2, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=wt)
ax.set_ylabel('wt')
ax.set_xlabel('am')
ax.set_title('am distribution against wt')

fig3, ax = plt.subplots(dpi=200)
sns.violinplot(x=am, y=qsec)
ax.set_ylabel('qsec')
ax.set_xlabel('am')
ax.set_title('am distribution against qsec')

fig4, ax = plt.subplots(dpi=200)
ax.plot(Js)

# univar logistic, contour map
theta, h, thetas = gradient_desc_logistic(am, drat_nor, alpha=0.01,
                                  convergence=1e-7, rec_theta = True,
                                  theta=[7.5, 15])
thetax, hx, thetasx = gradient_desc_logistic(am, drat_nor, alpha=0.01,
                                  convergence=1e-7, rec_theta = True,
                                  theta=[-8, -15])

theta_0 = np.linspace(-10,10,40)
theta_1 = np.linspace(-20,20,40)

theta_0, theta_1 = np.meshgrid(theta_0, theta_1)

X = make_X(drat_nor)
y = am

J_val = np.zeros((len(theta_0), len(theta_1)))
for i in range(0, len(theta_0)):
    for j in range(0, len(theta_1)):
        theta_map = np.array((theta_0[i,j], theta_1[i,j]))
        J_val[i,j] = Jcost(theta_map, X, y)
        
fig5, ax = plt.subplots(dpi=300)
ax.contour(theta_0, theta_1, J_val, levels=40, colors='black', linestyles='dashed', linewidths=0.8)
colored = ax.contourf(theta_0, theta_1, J_val, levels=40)

anno_steps = np.linspace(0, len(thetas)-1, 1000).tolist()
steps = [thetas[np.int(i)] for i in anno_steps]
for i in range(0, len(steps)-1):
    ax.annotate('', xy=steps[i+1], xytext=steps[i],
                arrowprops={'arrowstyle': '-', 'color': 'r', 'lw': 1},
                va='center', ha='center')
    
anno_stepsx = np.linspace(0, len(thetasx)-1, 1000).tolist()
stepsx = [thetasx[np.int(i)] for i in anno_stepsx]
for i in range(0, len(stepsx)-1):
    ax.annotate('', xy=stepsx[i+1], xytext=stepsx[i],
                arrowprops={'arrowstyle': '-', 'color': 'white', 'lw': 1},
                va='center', ha='center')

plt.colorbar(colored, shrink=0.8)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$', rotation = 1)
ax.set_title('Contour map of univariate gradient descend')


