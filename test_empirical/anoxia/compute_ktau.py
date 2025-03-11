#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:27:44 2020

Compute rolling ktau values from EWS dataframe

@author: Thomas M. Bury
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scipy.stats as stats


# Import EWS from forced thermoacoustic transitions
df_ews_forced = pd.read_csv('data/ews/df_ews_forced.csv')
df_ews_forced.set_index(['tsid','Variable label', 'Time'], inplace=True)

# Import EWS from null Dakos climate transitions
df_ews_null = pd.read_csv('data/ews/df_ews_null.csv')
df_ews_null.set_index(['tsid','Variable label', 'Time'], inplace=True)




# Function to compute kendall tau for time seires data up to point t_fin
def ktau_compute(series,t_fin):
    # selected data in series where from point where measured variable
    # is defined, up to t_fin
    t_start=series[pd.notnull(series)].index[1]
    
    series_reduced = series.loc[t_start:t_fin]
    x1 = series_reduced.index.values
    x2 = series_reduced.values
    ktau, pval = stats.kendalltau(x1,x2)
    return ktau

# Compute kendall tau values with spacing equal to spacing of ML predictions
def ktau_series(series):
    tVals = series.index[::10]
    ktauVals = []
    for t in tVals:
        ktau = ktau_compute(series,t)
        ktauVals.append(ktau)
    
    # Return series
    ktauSeries = pd.Series(ktauVals,index=tVals)
    return ktauSeries



#-------------
# Compute kendall tau for forced simulations
#------------

# Store list of dfs with kendall tau values from each simulation
list_df = []

# Loop through each time series ID
tsid_vals = df_ews_forced.index.unique(level='tsid')
for tsid in tsid_vals:
    for var in ['Mo','U']:

        series_var = df_ews_forced.loc[tsid,var]['Variance']
        series_ac = df_ews_forced.loc[tsid,var]['Lag-1 AC']
        
        # Compute kendall tau series
        series_ktau_var = ktau_series(series_var)
        series_ktau_var.name = 'ktau_variance'
        series_ktau_ac = ktau_series(series_ac)
        series_ktau_ac.name = 'ktau_ac'
    
        
        # Put into temporary dataframe
        df_temp = pd.concat([series_ktau_var, series_ktau_ac], axis=1).reset_index()
        df_temp['tsid'] = tsid
        df_temp['Variable label'] = var
        list_df.append(df_temp)
        
    print('K tau done for tsid {}'.format(tsid))
        

# Concatenate kendall tau dataframes
df_ktau_forced = pd.concat(list_df).set_index(['tsid','Time'])

# Export dataframe
df_ktau_forced.to_csv('data/ews/df_ktau_forced.csv')




# #-------------
# # Compute kendall tau for null simulations
# #------------

# Store list of dfs with kendall tau values from each simulation
list_df = []

# Loop through each time series ID
tsid_vals = df_ews_null.index.unique(level='tsid')
for tsid in tsid_vals:
    for var in ['Mo','U']:

        series_var = df_ews_null.loc[tsid,var]['Variance']
        series_ac = df_ews_null.loc[tsid,var]['Lag-1 AC']
        
        # Compute kendall tau series
        series_ktau_var = ktau_series(series_var)
        series_ktau_var.name = 'ktau_variance'
        series_ktau_ac = ktau_series(series_ac)
        series_ktau_ac.name = 'ktau_ac'
    
        
        # Put into temporary dataframe
        df_temp = pd.concat([series_ktau_var, series_ktau_ac], axis=1).reset_index()
        df_temp['tsid'] = tsid
        df_temp['Variable label'] = var
        list_df.append(df_temp)
        
    print('K tau done for tsid {}'.format(tsid))
        

# Concatenate kendall tau dataframes
df_ktau_null = pd.concat(list_df).set_index(['tsid','Time'])

# Export dataframe
df_ktau_null.to_csv('data/ews/df_ktau_null.csv')












