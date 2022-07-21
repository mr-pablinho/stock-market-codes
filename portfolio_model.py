# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:38:40 2021
PORTFOLIO MODEL - WHY GLOBAL SENSITIVITY?
@author: PMR

Example taken from:
- Smith, R.C., 2013. Uncertainty Quantification: Theory, Implementation, and Applications. SIAM.
"""


# %% Import libraries

import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt


# %% Portfolio model

def portfolio_model(c1, c2, Q1, Q2):
    Y = (c1 * Q1) + (c2 * Q2)
    # Q1 and Q2 hedged portfolios
    # c1 and c2 amounts invested 
    return Y


# %% Setup problem

Q1_mean = 0
Q1_std = 1
Q2_mean = 0
Q2_std = 3

c1 = 2
c2 = 1

Q1_distro = cp.Normal(Q1_mean, Q1_std)
Q2_distro = cp.Normal(Q2_mean, Q2_std)

np.random.seed(1)
nSamples = 1000
J_distro = cp.J(Q1_distro, Q2_distro)
samples = J_distro.sample(nSamples).T


# %% Evaluate the model

Y_all = []
for i in range(nSamples):
    Q1, Q2 = samples[i,0], samples[i,1]
    Y = portfolio_model(c1, c2, Q1, Q2)
    Y_all.append(Y)
    
    
# %% Plots

plt.figure('q1 v. y')
plt.scatter(samples[:,0], Y_all, s=5, color='blue', alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel('$q_1$')
plt.ylabel('$y$')
plt.xlim(-15,15)
plt.ylim(-15,15)

plt.figure('q2 v. y')
plt.scatter(samples[:,1], Y_all, s=5, color='blue', alpha=0.5)
plt.grid(alpha=0.3)
plt.xlabel('$q_2$')
plt.ylabel('$y$')
plt.xlim(-15,15)
plt.ylim(-15,15)
    
