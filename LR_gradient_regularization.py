# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:16:16 2021

@author: jacob
"""

import numpy as np
from numpy import linalg as lin
import pandas as pd
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



# extract theta_1 - theta_n for regularization use
def theta_no_cons_term(theta):
    new_theta = np.insert(theta[1:], 0, 0)
    return new_theta

# cost function
def Jcost(theta, X, y, lamb):
    m = len(y)
    h = X @ theta
    new_theta = theta_no_cons_term(theta)
    J = (1/(2*m)) * (np.sum((h - y)**2) + lamb*(np.sum(new_theta**2)))
    return J

# update theta
def gradient_update(X, y, theta, alpha, m, lamb):
    h = X @ theta
    # for theta_0
    theta[0] = theta[0] - (alpha/m)*(X.T @ (h-y))[0]
    # for theta_1 - theta_n
    theta[1:] = (1-(alpha*lamb)/m)*theta[1:] - (alpha/m)*(X.T @ (h-y))[1:]
    return theta


# gradient descending: set up
def gradient_desc(y_data, *x_data, alpha=0.01, lamb = 0, 
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
    J = Jcost(theta, X, y, lamb)

    # batch gradient descending
    while J_diff > convergence:
        update_theta = gradient_update(X, y, theta, alpha, m, lamb)
        update_J = Jcost(update_theta, X, y, lamb)
        J_diff = J - update_J
        if J_diff < 0:
            raise RuntimeError("Not converging! WRONG! TRY SMALLER ALPHA!")
        theta = update_theta
        J = update_J

    # calculate hypothesis h
    h = X @ theta
    
    return theta, h

#
# implementation
mpg, hp = choose_data(mtdata, 'mpg', 'hp')
mpg, hp = feat_scale(mpg, hp)

hp_sq = hp**2
hp_cb = hp**3

fig1, ax = plt.subplots(dpi = 300)
ax.scatter(y = mpg, x = hp)
ax.set_xlabel('hp')
ax.set_ylabel('mpg')


# straight line
theta_strt, h_strt = gradient_desc(mpg, hp)
 
ax.plot(hp, h_strt, color='red')

# quadratic without regularization
theta_quad, h_no_use = gradient_desc(mpg, hp, hp_sq)

x_hp = np.linspace(np.min(hp), np.max(hp), len(hp))
h_quad = theta_quad[0] + theta_quad[1]*x_hp + theta_quad[2]*(x_hp**2)

ax.plot(x_hp, h_quad, color='green', label='quadratic')

# cubic without regularization
theta_cubic, h_no_use= gradient_desc(mpg, hp, hp_sq, hp_cb)

h_cubic = theta_cubic[0] + theta_cubic[1]*x_hp + theta_cubic[2]*(x_hp**2)+theta_cubic[3]*(x_hp**3)

ax.plot(x_hp, h_cubic, color='orange', label='cubic')

plt.legend()
plt.show()

# cubic with regularization
theta_cb_reg, h_no_use = gradient_desc(mpg, hp, hp_sq, hp_cb, lamb=10)

h_cb_reg = theta_cb_reg[0] + theta_cb_reg[1]*x_hp + theta_cb_reg[2]*(x_hp**2)+theta_cb_reg[3]*(x_hp**3)

fig2, ax = plt.subplots(dpi=300)
ax.scatter(y = mpg, x = hp)
ax.set_xlabel('hp')
ax.set_ylabel('mpg')
ax.plot(x_hp, h_cubic, label='no reg')
ax.plot(x_hp, h_cb_reg, label='with reg')
plt.legend()


# finally: regulation with normal equation
def normal_equa(y_data, *x_data, lamb=0):
    X = make_X(*x_data)
    y = y_data
    reg_mat = np.eye(X.shape[1])
    reg_mat[0,0] = 0
    
    theta = lin.inv(X.T @ X + lamb*reg_mat) @ X.T @ y
    return theta

theta_res = normal_equa(mpg, hp, hp_sq, hp_cb, lamb=10)
    