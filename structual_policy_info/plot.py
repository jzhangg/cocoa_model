#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:18:31 2022

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

from functions import cocoa, draw_df, cocoa_cal

path = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%

# define plot functions 

## display

## Optimal F under no subsidy, GPP or PES 

sns.set(rc={'axes.facecolor':'lavender'})

# -1- (histogram of F_star) 2 by 3: no gamma variance
titles = np.array([[i+ ' , $\gamma=$'+str(j) for i in ["No program","PES", "GPP"]] for j in [3, 1.2] ])
plot_opt = [ [[i,j] for j in ['F_non', 'F_pes', 'F_gpp']] for i in [cocoa_gamma_H, cocoa_gamma_L]]
fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 8), sharey = True, sharex = True)
for i in range(2):
    for j in range(3):
        df = plot_opt[i][j][0]
        v  = plot_opt[i][j][1]      
        axs[i,j].hist(df[v], color = 'green', alpha = 0.5, bins = 25)
        axs[i,j].set_title(titles[i,j])
axs[1,1].set_xlabel("Shade level F")
axs[0,0].set_ylabel(r'$\theta$')     
axs[1,0].set_ylabel(r'$\theta$')     
plt.show()
fig.savefig('../output/'+ 'hist_optF_gamma.png')


## -2- (scatter of p and theta, colored by F_star) 2 by 3: no gamma variance
plot_opt = [ [[i,j] for j in ['F_non', 'F_pes', 'F_gpp']] for i in [cocoa_gamma_H, cocoa_gamma_L]]
fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 8), sharey = True, sharex = True)
for i in range(2):
    for j in range(3):
        df = plot_opt[i][j][0]
        v  = plot_opt[i][j][1]      
        axs[i,j].scatter(df["p"], df["theta"], s = 20, alpha = 0.5, c = df[v], cmap='Greens')
        axs[i,j].set_title(titles[i,j])
axs[1,1].set_xlabel("p")
axs[0,0].set_ylabel(r'$\theta$')     
axs[1,0].set_ylabel(r'$\theta$')     
plt.show()
fig.savefig('../output/'+ 'optF_gamma.png')



# var(gamma) 1& 2
dfs = [cocoa_gamma1, cocoa_gamma2, cocoa_gamma3, cocoa_gamma4, cocoa_gamma5, cocoa_chi2]   
titles = np.array(["No program", "PES", "GPP"])
plot_opt = np.array(['F_non', 'F_pes', 'F_gpp'])

# -1- (histogram of F_star)
for k, data in enumerate(dfs): 
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15,6), sharey = True, sharex = True)
    for i in range(3):
        df = data.copy()
        v  = plot_opt[i]      
        axs[i].hist(df[v], color = 'green', alpha = 0.5, bins = 25)
        axs[i].set_title(titles[i])
        axs[i].set_xlim(0,1)
    axs[0].set_ylabel(r'$\theta$') 
    axs[1].set_xlabel("Shade level F")   
    plt.show()
    fig.savefig('../output/' + 'hist_optF_compare_' + str(k) + '.png')
    

#  -2- (scatter of p and theta, colored by F_star)
dfs = [cocoa_gamma1, cocoa_gamma2, cocoa_gamma3, cocoa_gamma4, cocoa_gamma5, cocoa_chi2]   
plot_opt = np.array(['F_non', 'F_pes', 'F_gpp'])
for k, data in enumerate(dfs): 
    df = data.copy()
    df['index'] = df.index
    df_opt_F = pd.wide_to_long(df[['index', 'p', 'theta', 'gamma', 'F_non', 'F_pes', 'F_gpp']], stubnames = 'F', 
                             i = 'index', j ='Subsidy_type', sep = '_', suffix=r'\w+' )
    g = sns.relplot(
        data=df_opt_F, x="p", y="theta",
        col= 'Subsidy_type', hue="F", palette='Greens',
        kind="scatter").set(ylabel = r'$\theta$') 
    for i in range(3):
        g.fig.axes[i].set_title(titles[i])
    plt.show()    
    g.savefig('../output/' + 'optF_compare_' + str(k) + '.png')


#%%
## Take-up

## -1- Scatter plot of p and theta, colored by takeup. no gamma variance 
titles = np.array([[i+ ' , $\gamma=$'+str(j) for i in ["PES", "GPP"]] for j in [3, 1.2] ])
plot_opt = [ [[i,j] for j in ['PES_takeup', 'GPP_takeup']] for i in [cocoa_gamma_H, cocoa_gamma_L]]
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 8), sharey = True, sharex = True)
for i in range(2):
    for j in range(2):
        df = plot_opt[i][j][0]
        v  = plot_opt[i][j][1]      
        axs[i,j].scatter(df[df[v]==1]["p"], df[df[v]==1]["theta"], s = 12,  alpha=0.5, label= "True")
        axs[i,j].scatter(df[df[v]==0]["p"], df[df[v]==0]["theta"], s = 12,  alpha=0.5, label= "False")
        axs[i,j].set_title(titles[i,j])
axs[1,1].set_xlabel("p")
axs[1,0].set_xlabel("p")
axs[0,0].set_ylabel(r'$\theta$')     
axs[1,0].set_ylabel(r'$\theta$')     
plt.show()
fig.savefig('../output/' + 'scatter_takeup_gamma.png')


## -2- scatter plot of p and theta, colored by takeup. vary gamma 
dfs = [cocoa_gamma1, cocoa_gamma2, cocoa_gamma3, cocoa_gamma4, cocoa_gamma5, cocoa_chi2]   
titles = np.array(["PES", "GPP"])
plot_opt = np.array(['PES_takeup', 'GPP_takeup'])
# hist
for k, data in enumerate(dfs): 
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(10,5), sharey = True, sharex = True)
    for i in range(2):
        df = data.copy()
        v  = plot_opt[i] 
        axs[i].scatter(df[df[v]==1]["p"], df[df[v]==1]["theta"], s = 25,  alpha=0.3, label= "True")
        axs[i].scatter(df[df[v]==0]["p"], df[df[v]==0]["theta"], s = 25,  alpha=0.3, label= "False")
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("p")     
    axs[0].set_ylabel(r'$\theta$')
    axs[0].set_ylim(0.05, 0.4)
    axs[0].set_xlim(0, 0.7)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = "upper center" , ncol = 2)
    plt.show()
    fig.savefig('../output/' + 'scatter_takeup_compare_' + str(k) + '.png')



## save PES, GPP separately
# -1- scatter plot of p and theta, colored by takeup
for i in ["PES", "GPP"]:
    df = cocoa_gamma1.copy()
    v   = i + "_takeup"
    title  = i + " takeup"
    fig, ax = plt.subplots(figsize=(6,6))
    plt.scatter(df[df[v]==1]["p"], df[df[v]==1]["theta"], s = 30,  alpha=0.3, linewidth= 0,  label= "True")
    plt.scatter(df[df[v]==0]["p"], df[df[v]==0]["theta"], s = 30,  alpha=0.3, linewidth= 0, label= "False")
    plt.title(title)
    plt.xlabel('p')
    plt.ylabel(r'$\theta$')
    plt.legend(title="Takeup", loc = "upper right", fontsize = 10)
    plt.show()
    fig.savefig('../output/'+ i + '_scatter_takeup.png')
    print(i + 'done')
    
# Gamma distribution 
# -2- histogram of gamma
for i in ["PES", "GPP"]:
    hues   = i + "_takeup"
    g = sns.histplot(data=cocoa_gamma1, x="gamma", alpha = 0.5, bins= 25,
                 hue = hues, hue_order=[True, False], multiple="stack")
    g.set_ylabel('Frequency')
    g.set_xlabel(r'$\gamma$')
    plt.show()
    g.figure.savefig('../output/'+ i + '_hist_gamma.png')


## Delta F under GPP or PES

# Two levels of gamma (2 by 2): GPP/PES
# -1- scatter of p and theta, colored by deltaF
titles = np.array([[i+ ' , $\gamma=$'+str(j) for i in ["PES", "GPP"]] for j in [3, 1.2] ])
plot_opt = [ [[i,j] for j in ['deltaF_pes', 'deltaF_gpp']] for i in [cocoa_gamma_H, cocoa_gamma_L]]
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 12), sharey = True, sharex = True)
for i in range(2):
    for j in range(2):
        df = plot_opt[i][j][0]
        v  = plot_opt[i][j][1]   
        g = sns.scatterplot(data = df[df[v] ==0], x="p", y="theta", alpha = 0.5, 
                            color = "gray",ax = axs[i,j]).set(ylabel=r"$\theta$", title=titles[i,j])
        sns.scatterplot(data = df[df[v] < 0], x="p", y="theta", hue = v, palette='Oranges_r', ax = axs[i,j])       
        sns.scatterplot(data = df[df[v] > 0], x="p", y="theta", hue = v, palette='Greens',  ax = axs[i,j])
plt.show()
fig.savefig('../output/'+ 'scatter_deltaF_gamma.png')


# Delta F: varying gamma (1 by 2)
# -2- scatter of  p and theta, colored by deltaF
dfs = [cocoa_gamma1, cocoa_gamma2, cocoa_gamma3, cocoa_gamma4, cocoa_gamma5, cocoa_chi2]   
titles = np.array(["PES", "GPP"])
plot_opt = np.array(['deltaF_pes', 'deltaF_gpp'])
# hist
for k, data in enumerate(dfs): 
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(10,5), sharey = True, sharex = True)
    for i in range(2):
        df = data.copy()
        v  = plot_opt[i] 
        print(v)
        g = sns.scatterplot(data = df[df[v] ==0], x="p", y="theta", alpha = 0.5, 
                            color = "gray",ax = axs[i]).set(ylabel=r"$\theta$", title=titles[i])
        sns.scatterplot(data = df[df[v] < 0], x="p", y="theta", hue = v, palette='Oranges_r', ax = axs[i])       
        sns.scatterplot(data = df[df[v] > 0], x="p", y="theta", hue = v, palette='Greens',  ax = axs[i])
    axs[0].set_ylim(0.05, 0.4)
    axs[0].set_xlim(0, 0.7)
    plt.show()
    fig.savefig('../output/' + 'scatter_deltaF_compare_'+str(k) + '.png')


## joinplots: joint F(theta,p; gamma) with marginal histogram
dfs = [cocoa_gamma_H, cocoa_gamma_L, cocoa_gamma1, cocoa_gamma2, cocoa_gamma3, cocoa_gamma4, cocoa_gamma5, cocoa_chi2]   
index = ["H", "L", "0", "1", "2", "3", '4', '5']
for j, dfi in enumerate(dfs):
    ind_j = index[j]
    for i in ['PES','GPP']:
        hues   = i + "_takeup"
        title  = i + " takeup"
        g = sns.jointplot(data=dfi, 
                  x="p", y="theta", hue = hues, hue_order = [True, False],
                  ratio = 3, kind='hist', alpha = 0.4, palette= 'tab10', bins = 25, marginal_ticks=True,
                  marginal_kws={ 'palette': 'tab10', 'alpha': 0.4, 
                                'multiple': 'stack',  'bins': 25, 'hue_order': [True, False]})
        g.ax_joint.set_xlabel(r"$p$")
        g.ax_joint.set_ylabel(r"$\theta$ ")
        g.ax_marg_x.set_xlim(0, 0.7)
        g.ax_marg_y.set_ylim(0.05, 0.4)
        g.fig.suptitle(title)
        plt.show()
        g.fig.savefig('../output/'+ i + '_takeup_gamma' + ind_j + ".png" )
        print(i + ind_j + 'done')
    
    
    
    


## Scatter plot: (gamma, p)
plt.figure(figsize=(8,8))
plt.scatter(df.p[df.GPP_takeup==1], df.gamma[df.GPP_takeup==1], s=10, alpha=0.5)
plt.scatter(df.p[df.GPP_takeup==0], df.gamma[df.GPP_takeup==0], s=10, alpha=0.5)
plt.title('GPP takeup (blue)')
plt.xlabel('p')
plt.ylabel(r'$\gamma$')
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(df.p[df.PES_takeup==1], df.gamma[df.PES_takeup==1], s=10, alpha=0.5)
plt.scatter(df.p[df.PES_takeup==0], df.gamma[df.PES_takeup==0], s=10, alpha=0.5)
plt.title('PES takeup (blue)')
plt.xlabel('p')
plt.ylabel(r'$\gamma$')
plt.show()











