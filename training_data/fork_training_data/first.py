#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas M. Bury

Simulate May's harvesting model 
Simulations going through Fold bifurcation
Compute EWS

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ewstools


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 0.01
t0 = 0
tmax = 1500 # for 1500-classifier
tburn = 100 # burn-in period
numSims = 10
seed = 0 # random number generation seed
sigma = 0.01 # noise intensity

# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
span = 0.2 # bandwidth
lags = [1] # autocorrelation lag times
ews = ['var','ac']


#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(x,r,a):
    return r*x +a*x**3

#a=+-1
    
# Model parameters
a=1
r=1
#Il faut ajouter termes ordres sup
#ajouter bruit blanc gaussien
#gérer format sorite #DUR


# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))

# Set bifurcation parameter b, that increases linearly in time from bl to bh
b = pd.Series(np.linspace(bl,bh,len(t)),index=t)
# Time at which bifurcation occurs
tcrit = b[b > bcrit].index[1]

## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun(x0,r,k,b[0],s)*dt + dW_burn[i]
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i],r,k, b.iloc[i], s)*dt + dW[i]
        # make sure that state variable remains >= 0
        if x[i+1] < 0:
            x[i+1] = 0
            
    # Store series data in a temporary DataFrame
    data = {'tsid': (j+1)*np.ones(len(t)),
                'Time': t,
                'x': x}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation '+str(j+1)+' complete')

#  Concatenate DataFrame from each tsid
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['tsid','Time'], inplace=True)



#----------------------
# Compute EWS for each tsid 
#---------------------

# Filter time-series to have time-spacing dt2
df_traj_filt = df_traj.loc[::int(dt2/dt)]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []
appended_pspec = []

# loop through tsid
print('\nBegin EWS computation\n')
for i in range(numSims):
    # loop through variable (only 1 in this model)
    for var in ['x']:
        
        ews_dic = ewstools.core.ews_compute(df_traj_filt.loc[i+1][var], 
                          roll_window = rw,
                          smooth='Lowess',
                          span = span,
                          lag_times = lags, 
                          ews = ews,
                          upto=tcrit)
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
        
        # Include a column in the DataFrames for realisation number and variable
        df_ews_temp['tsid'] = i+1
        df_ews_temp['Variable'] = var
        
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
        
    # Print status every realisation
    if np.remainder(i+1,1)==0:
        print('EWS for realisation '+str(i+1)+' complete')


# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(appended_ews).reset_index().set_index(['tsid','Variable','Time'])


# #-------------------------
# # Plots to visualise EWS
# #-------------------------

# # Realisation number to plot
# plot_num = 4
# var = 'x'
# ## Plot of trajectory, smoothing and EWS of var (x or y)
# fig1, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,6))
# df_ews.loc[plot_num,var][['State variable','Smoothing']].plot(ax=axes[0],
#           title='Early warning signals for a single realisation')
# df_ews.loc[plot_num,var]['Variance'].plot(ax=axes[1],legend=True)
# df_ews.loc[plot_num,var]['Lag-1 AC'].plot(ax=axes[2],legend=True)



#------------------------------------
# Export data 
#-----------------------------------

# Export EWS data
df_ews.to_csv('data/ews/df_ews_forced.csv')

# Export residuals as individual files for training ML
for i in np.arange(numSims)+1:
    df_resids = df_ews.loc[i,'x'].reset_index()[['Time','Residuals']]
    filepath='data/resids/pitchfork_1500_resids_{}.csv'.format(i)
    df_resids.to_csv(filepath,
                      index=False)

