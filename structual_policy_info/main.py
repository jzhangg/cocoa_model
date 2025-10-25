#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:16:56 2022

@author: Jac
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import Bounds

import multiprocessing as mp
from time import time


path = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from functions import cocoa, draw_df, cocoa_cal

#%%


## set up the parameters
Yd = 0.5
Y0 = 1
C0 = 0.08
C1 = 0.08
Fbar = 0.6
theta_range = [C0, Y0-Yd]  
#subsidy = [0.15, 0.14] # [delta, alpha]
subsidy = [0.1, 0.1] # [delta, alpha]
N = 2000

#%%
# Case 1a: gamma high, no var 
mu      = (0.2, 0.2, 3)
var     = np.diag([0.01, 0.0025, 0])   
df_gamma_H = draw_df(mu, var, N, theta_range, subsidy)
cocoa_gamma_H, W_gamma_H, trial_H = cocoa_cal(df_gamma_H, Y0, Yd, C0, C1, Fbar)

# Case 1b: gamma low, no var
mu      = (0.2, 0.2, 1.2)
var     = np.diag([0.01, 0.0025, 0])   
df_gamma_L = draw_df(mu, var, N, theta_range, subsidy)
cocoa_gamma_L, W_gamma_L, trial_L = cocoa_cal(df_gamma_L, Y0, Yd, C0, C1, Fbar)

# Case 2a: gamma add variation
mu      = (0.2, 0.2, 2)
var     = np.diag([0.01, 0.0025, 1])   
df_gamma1 = draw_df(mu, var, N, theta_range, subsidy)
cocoa_gamma1, W_gamma1, trial1 = cocoa_cal(df_gamma1, Y0, Yd, C0, C1, Fbar)

#%%
# Case 2b: a more concentrated distribution
mu      = (0.2, 0.2, 2)
var     = np.diag([0.0025, 0.0001, 1])   
df_gamma2 = draw_df(mu, var, N, theta_range, subsidy)
cocoa_gamma2, W_gamma2, trial2 = cocoa_cal(df_gamma2, Y0, Yd, C0, C1, Fbar)


# Case 2c: average low theta --> PES can be better if people have a low theta overall
mu      = (0.4, 0.13, 2)
var     = np.diag([0.01, 0.0025, 1])   
df_gamma3 = draw_df(mu, var, N, theta_range, subsidy)
cocoa_gamma3, W_gamma3, trial3 = cocoa_cal(df_gamma3, Y0, Yd, C0, C1, Fbar)

#%% Impact of correlation

# Case 4: positive correlation (0.8)
mu      = (0.2, 0.2, 2)
cov     = np.array([[0.01,   0.004,    0],
                   [0.004,    0.0025, 0],
                   [0,      0,      1]])   
df_gamma4 = draw_df(mu, cov, N, theta_range, subsidy)
cocoa_gamma4, W_gamma4, trial4 = cocoa_cal(df_gamma4, Y0, Yd, C0, C1, Fbar)


# Case 5: negative correlation (-0.8)
mu      = (0.2, 0.2, 2)
cov     = np.array([[0.01,   -0.004,    0],
                   [-0.004,    0.0025, 0],
                   [0,      0,      1]])   
df_gamma5 = draw_df(mu, cov, N, theta_range, subsidy)
cocoa_gamma5, W_gamma5, trial5 = cocoa_cal(df_gamma5, Y0, Yd, C0, C1, Fbar)

#%% Chi square distribution

mu      = (0.2, 0.2, 2)
var     = np.diag([0.0025, 0.0001, 1])   
df_chi2 = draw_df(mu, var, N, theta_range, subsidy, method = 'chi2')
cocoa_chi2, W_chi2, trial_chi2 = cocoa_cal(df_chi2, Y0, Yd, C0, C1, Fbar)
