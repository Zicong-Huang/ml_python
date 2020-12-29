# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:04:08 2020

@author: Zicong Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data
mtdata = pd.read_csv("mtcars.csv")


# prepare variables
mpg = mtdata.loc[:,"mpg"].to_numpy()   # miles per gallon
wt = mtdata.loc[:,"wt"].to_numpy()     # weight of the car
hp = mtdata.loc[:,"hp"].to_numpy()     # horsepower 

# create scatter plot
fig1, ax = plt.subplots(dpi = 300)
ax.scatter(mpg, hp, marker='x', color='red')


# feature scaling
def feat_scale(feature):
    return (feature - feature.mean()) / (feature.std())


# compose the desired matrix
def make_X(x_data):
    m = len(x_data)
    return np.stack((np.ones(m), x_data)).T


# implement the cost function
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
def univ_gradient_desc(y_data, x_data, alpha, convergence=1e-7, theta=[0,0], record_theta=False):
    # set up
    m = len(y_data)
    X = make_X(x_data)
    y = y_data.T
    theta = np.array(theta)
    # initial cost
    J_diff = 10
    J = Jcost(theta, X, y)

    # batch gradient descending, default
    if record_theta==False:
        while J_diff > convergence:
            theta = gradient_update(X, y, theta, alpha, m)
            update_J = Jcost(theta,X,y)
            J_diff = J - update_J
            if J_diff < 0:
                raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
            J = update_J
    else:
        thetas = list()
        while J_diff > convergence:
            thetas.append(theta.tolist())
            theta = gradient_update(X, y, theta, alpha, m)
            update_J = Jcost(theta,X,y)
            J_diff = J - update_J
            if J_diff < 0:
                raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
            J = update_J

    # calculate hypothesis h
    h = np.matmul(X, theta)
    
    if record_theta == False: return theta, h
    else:
        return thetas

#
# implementation
mpg_nor = feat_scale(mpg)
wt_nor = feat_scale(wt)
hp_nor = feat_scale(hp)

theta, h = univ_gradient_desc(mpg_nor, hp_nor, alpha = 0.01)

#
# plot the fitted lines
fig2,ax = plt.subplots(dpi=300)
ax.plot(hp_nor, h)
ax.scatter(hp_nor,mpg_nor, marker='x', color='red')

#
# contour graph
thetas = univ_gradient_desc(mpg_nor, hp_nor, alpha = 0.1, theta=[0.5, 0.5], record_theta=True)

theta_0 = np.linspace(-1,1,20)
theta_1 = np.linspace(-1,1,20)

theta_0, theta_1 = np.meshgrid(theta_0, theta_1)

X = make_X(hp_nor)
y = mpg_nor

J_val = np.zeros((len(theta_0), len(theta_1)))
for i in range(0, len(theta_0)):
    for j in range(0, len(theta_1)):
        theta = np.array((theta_0[i,j], theta_1[i,j]))
        J_val[i,j] = Jcost(theta, X, y)

fig3,ax = plt.subplots(dpi=300)
ax.contour(theta_0, theta_1, J_val, levels=30, colors='black', linestyles='dashed', linewidths=1)
colored = ax.contourf(theta_0, theta_1, J_val, levels=30)
anno_steps = np.linspace(0, len(thetas)-1, 7).tolist()
steps = [thetas[np.int(i)] for i in anno_steps]
for i in range(0, len(steps)-1):
    ax.annotate('', xy=steps[i+1], xytext=steps[i],
                arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                va='center', ha='center')
plt.colorbar(colored, shrink=0.8)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$', rotation = 1)
ax.set_title('Contour map of univariate gradient descend')

#
# descending through iterations
thetas = univ_gradient_desc(mpg_nor, hp_nor, alpha = 0.1, theta=[0.5, 0.5], record_theta=True)
J_rec = np.zeros(len(thetas))
for i in range(0, len(J_rec)):
    J_rec[i] = Jcost(thetas[i], X, y)

fig4,ax = plt.subplots(dpi=300)
ax.plot(J_rec)
ax.set_xlabel('number of iterations')
ax.set_ylabel('cost function values')
ax.set_title('Cost function value over iterations')

