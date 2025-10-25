#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:19:45 2022

@author: Jac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import Bounds

import multiprocessing as mp
from time import time

#%%




class cocoa:
    
    def __init__(self, data, Y0=1, Yd=0.5, C0=0.1, C1=0.1, Fbar = 0.5):
        self.data   = data.to_numpy()
        self.Y0     = Y0
        self.Yd     = Yd
        self.C0     = C0
        self.C1     = C1
        self.Fbar   = Fbar
        self.p      = data['p']
        self.theta  = data['theta']
        self.alpha  = data['alpha']
        self.gamma  = data['gamma']
        self.delta  = data['delta']
        self.N      = data.shape[0]
        
    def U(self, x, gamma):
        u = (x**(1-gamma)-1)/(1-gamma)
        return u
    
    def Up(self,x,gamma):
        return x**(-gamma)
    
    def _U_non(self, F, p, theta, gamma):
        U1 = self.U(self.Y0 - self.C0 * F , gamma )
        U2 = self.U(self.Yd - (self.C0 - theta)*F , gamma)
        EU = (1-p)*U1 + p*U2
        return EU
    
    def _U_pes(self, F, p, theta, gamma, delta):
        I = F >= self.Fbar
        U1 = self.U(self.Y0 - self.C1 - self.C0 * F + delta*I , gamma )
        U2 = self.U(self.Yd - self.C1 - (self.C0 - theta)*F + delta*I, gamma )
        EU = (1-p)*U1 + p*U2
        return EU
    
    def _U_gpp(self, F, p, theta, gamma, alpha):
        #K  = 1 + alpha * np.log( 1 + F )
        #K  = 1 + alpha * np.sqrt( F )
        K = 1 + alpha * F
        U1 = self.U(self.Y0 * K - self.C1 - self.C0 * F, gamma )
        U2 = self.U((self.Yd + theta*F ) * K - self.C1 - self.C0 * F, gamma )
        EU = (1-p)*U1 + p*U2
        return EU
    
        
    
    
    def F_non(self):
        obj = lambda x,p,theta,gamma: -self._U_non(x, p, theta, gamma)
        #bounds = Bounds(0, 1)
        #F0 = np.array([minimize(obj, x0 = xi, args=(pi,thetai, gammai), method='trust-constr', bounds=bounds).x[0] 
        #              for xi,pi,thetai,gammai in zip(self.p, self.p, self.theta, self.gamma)])
        F0 = np.array([minimize(obj, x0 = xi, args=(pi,thetai, gammai), method='Nelder-Mead').x[0] 
                       for xi,pi,thetai,gammai in zip(self.p, self.p, self.theta, self.gamma)])
        F0 = np.maximum(F0,0)
        F0 = np.minimum(F0,1)
        self.F_non_val = F0
        return F0

    def F_pes(self):
        obj = lambda x,p,theta,gamma,delta: -self._U_pes(x, p, theta, gamma, delta)
        F1 = np.array([minimize(obj, x0 = xi, args=(pi,thetai, gammai, deltai), method='Nelder-Mead').x[0] 
                       for xi,pi,thetai,gammai, deltai in zip(self.p, self.p, self.theta, self.gamma, self.delta)])
        F1 = np.maximum(F1,0)
        F1 = np.minimum(F1,1)
        self.F_pes_val = F1
        return F1
    
    def F_gpp(self):
        obj = lambda x,p,theta,gamma,alpha: -self._U_gpp(x, p, theta, gamma, alpha)
        F2 = np.array([minimize(obj, x0 = xi, args=(pi,thetai, gammai, alphai), method='Nelder-Mead').x[0] 
                       for xi,pi,thetai,gammai, alphai in zip(self.p, self.p, self.theta, self.gamma, self.alpha)])
        F2 = np.maximum(F2,0)
        F2 = np.minimum(F2,1)
        self.F_gpp_val = F2
        return F2
    
    
    
    def U_non_star(self):
        return self._U_non(self.F_non_val, self.p, self.theta, self.gamma)
    
    def U_pes_star(self):
        return self._U_pes(self.F_pes_val, self.p, self.theta, self.gamma, self.delta)
    
    def U_gpp_star(self):
        return self._U_gpp(self.F_gpp_val, self.p, self.theta, self.gamma, self.alpha)
    
    
    
    def PES_takeup(self):
        self.Takeup_pes = self.U_pes_star() > self.U_non_star()
        return self.Takeup_pes
    
    def GPP_takeup(self):
        self.Takeup_gpp = self.U_gpp_star() > self.U_non_star()
        return self.Takeup_gpp
    
    
    def PES_spending(self):
        self.subsidy_pes = self.delta  * self.Takeup_pes
        spending = np.sum(self.subsidy_pes - self.C1)
        return spending, spending
    
    def GPP_spending(self):
        self.subsidy_gpp_drou = self.Takeup_gpp*( (self.Yd+self.theta*self.F_gpp_val) * (self.alpha*self.F_gpp_val))
        self.subsidy_gpp_norm = self.Takeup_gpp*(self.Y0*(self.alpha*self.F_gpp_val)) 
                                                 
        spending_norm = np.sum(self.subsidy_gpp_norm - self.Takeup_gpp*self.C1)
        spending_drou = np.sum(self.subsidy_gpp_drou - self.Takeup_gpp*self.C1)
        return spending_norm, spending_drou
    
    def PES_extraF(self):
        return np.sum(self.Takeup_pes*(self.F_pes_val - self.F_non_val))
        
    def GPP_extraF(self):
        return np.sum(self.Takeup_gpp*(self.F_gpp_val - self.F_non_val))
    
    
    
    
#%% Simulation

def draw_df(mean, cov, N, cons, premia, method = 'normal'):
    theta_lb  = cons[0]
    theta_ub  = cons[1]
    if method == 'normal' :
        randnum = np.random.multivariate_normal(mean, cov, size=N)
        randnum[randnum<0] = 1e-10
    if method == 'chi2':
        randnum = np.random.chisquare(3,(N,3)) * mean / 3
    
    df = pd.DataFrame(data = randnum, 
              columns = ["p",
                         "theta", 
                         "gamma"])
    df['p'][df['p']>1] = 1-1e-10
    df['theta'][df['theta']>theta_ub] = theta_ub
    #df['theta'][df['theta']<theta_lb] = theta_lb 
    df['gamma'][df['gamma']==1] = 1-1e-10
    df['delta'] = premia[0]
    df['alpha'] = premia[1]
    return df
    

def cocoa_cal(data, Y0, Yd, C0, C1, Fbar):
    start_time = time()
    
    df = data.copy()
    trial = cocoa(df, Y0, Yd, C0, C1, Fbar)
    # No-program
    df['F_non'] = trial.F_non()
    df['U_non'] = trial.U_non_star()
    # PES
    df['F_pes'] = trial.F_pes()
    df['U_pes'] = trial.U_pes_star()
    df['PES_takeup'] = trial.PES_takeup()
    #GPP
    df['F_gpp'] = trial.F_gpp()
    df['U_gpp'] = trial.U_gpp_star()
    df['GPP_takeup'] = trial.GPP_takeup()
    # Delta F
    df['deltaF_pes'] = (df['F_pes']-df['F_non'])*df['PES_takeup']
    df['deltaF_gpp'] = (df['F_gpp']-df['F_non'])*df['GPP_takeup']
    
    
    
    # print stats
    values = [np.sum(trial.Takeup_pes),
          np.sum(trial.Takeup_gpp),
          trial.PES_extraF(),
          trial.GPP_extraF(),
          np.sum(trial.F_non_val),
          trial.PES_extraF()/np.sum(trial.F_non_val),          
          trial.GPP_extraF()/np.sum(trial.F_non_val),
          trial.PES_spending(),
          trial.GPP_spending()]
                                    
    terms = ["Number of takeups under PES:",
         "Number of takeups under GPP:",
         "Net increase in F under PES:",
         "Net increase in F under GPP:",
         "total F under None", 
         "df_PES",
         "df_GPP",
         "Net spending for PES:",
         "Net spending for GPP:"]
    welfare = pd.DataFrame({'Terms': terms, 
                          'Value': values})
    
    
    # Subsidy received
    df['subs_pes'] = trial.subsidy_pes
    df['subs_gpp_norm'] = trial.subsidy_gpp_norm
    df['subs_gpp_drou'] = trial.subsidy_gpp_drou
    
    
    print("--- %s seconds ---" % (time() - start_time))
    
    return df, welfare, trial
    



    