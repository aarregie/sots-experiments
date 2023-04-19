# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:26:19 2023

@author: aarregui
"""


import sys
import os
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (45,17)
from src.utils import config_parser, directories
from src.utils.instants import *
import pmdarima as pm
import logging
from src.utils.cycle_selection import delete_anomalous_cycles
import seaborn as sns

sns.set_theme(style='white', font_scale = 6)
#%%

cfg_file = os.path.join('cfg', 'cfg.yaml')
config = config_parser.parse_config(cfg_file)


output_folder = config['output_folder']
interpolators_folder = config['interpolators_folder']
predictions_folder = config['predictions_folder']

cell_path = config['data_path']

#list data

cells = os.listdir(cell_path)

variables = ['I', 'Qc', 'Qd', 'T', 'V']

sum_test = 0

max_test_length = 0
max_test = np.array([])

cell_name = config['cell']



cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
with open(cell_file,'rb') as file:
    cell = pickle.load(file)
file.close()

    
s = config['s']
c = config['c']

cqd = config['var']


cycles = list(cell[c].keys())

duration_list = list()

for cycle in cycles:
    
    
    data = cell[c][cycle]['t']
    
    duration = data.max()
    
    duration_list.append(duration)

    

max_duration = np.max(duration_list)

instants_array = np.linspace(0, max_duration, 50)



#probar plot
analysis_UD = ['ARIMA', 'GP']
analysis_MD = ['MTGP', 'LRVAR', 'BVAR']


y_true = cell[c]['100'][cqd]
t_true = cell[c]['100']['t']

duration_cycle = t_true.max()
n_obs = sum(instants_array<duration_cycle)

interval_segmentation = np.linspace(0, duration_cycle, n_obs)
t_true_new = np.linspace(0, duration_cycle, len(t_true))

for analysis in analysis_UD:
    hat_file = os.path.join('output', cell_name, 'univariate', analysis, 'T', 'predictions', 'pred_T_C100_T50.csv')
    
    pred = pd.read_csv(hat_file)
    y_hat = pred['y_hat'].to_numpy()
    t_hat = pred['time'].to_numpy()
    
    example_hat_file = os.path.join('output', cell_name, 'meta', '{}_example_yhat.csv'.format(analysis))
    
    example_pred = pd.read_csv(example_hat_file)['y_hat'].to_numpy()
    
    adding_time = instants_array[n_obs-1:]
    adding_pred = np.append(np.array([y_hat[-1]]), example_pred[:-n_obs])
    
    cmask = np.zeros(len(t_true))
    
    for instant in instants_array:

        if instant <= duration_cycle and instant != 0:
            diff_instants = abs(t_true_new - instant)
            mask_idx = np.argmin(diff_instants)
            
            cmask[mask_idx] = 1
        if instant == 0:
            
            cmask[0] = 1
            
    cmask_bool = cmask == 1
    
    
    fig = plt.figure()
    l = sns.lineplot(x = t_true_new, y = y_true, markevery = cmask_bool, marker = 'D', markersize = 18, lw = 5)
    sns.lineplot(x = instants_array[:n_obs], y = y_hat,marker = 'o', markersize = 18, ls = (0, (5,5)), lw = 6, color = 'orange')
    sns.lineplot(x = adding_time, y = adding_pred, marker = 'o', markersize = 18, ls = (0,(5,5)), lw = 6, color = 'orange', alpha = .5)
    l.set_xticks(np.linspace(0, instants_array.max(), 6))
    l.set_xticklabels(np.array([1, 10, 20, 30, 40, 50]))
    
    plt.vlines(x = instants_array, ymin = 32, ymax = 40, color = 'grey', alpha = .3)
    plt.axvline(x = duration_cycle, color = 'red', ls = 'dotted', lw = 5) 
    plt.ylabel('')
    plt.legend(labels = ['$C_{100}$', '$\hat{C}_{100}$'], loc = 'lower left')

    plt_name = os.path.join('output', cell_name, 'meta', '{}_example.png'.format(analysis))
    plt.savefig(plt_name)
    
    plt.show()


#%%
for analysis in analysis_MD:
    hat_file = os.path.join('output', cell_name, 'multivariate', analysis, 'T', 'predictions', 'pred_T_C100_T50.csv')
    
    pred = pd.read_csv(hat_file)
    y_hat = pred['y_hat'].to_numpy()
    t_hat = pred['time'].to_numpy()
    
    example_hat_file = os.path.join('output', cell_name, 'meta', '{}_example_yhat.csv'.format(analysis))
    
    example_pred = pd.read_csv(example_hat_file)['y_hat'].to_numpy()
    
    adding_time = instants_array[n_obs-1:]
    adding_pred = np.append(np.array([y_hat[-1]]), example_pred[:-n_obs])
    
    cmask = np.zeros(len(t_true))
    
    for instant in instants_array:

        if instant <= duration_cycle and instant != 0:
            diff_instants = abs(t_true_new - instant)
            mask_idx = np.argmin(diff_instants)
            
            cmask[mask_idx] = 1
        if instant == 0:
            
            cmask[0] = 1
            
    cmask_bool = cmask == 1
    
    
    fig = plt.figure()
    l = sns.lineplot(x = t_true_new, y = y_true, markevery = cmask_bool, marker = 'D', markersize = 18, lw = 6)
    sns.lineplot(x = instants_array[:n_obs], y = y_hat,marker = 'o', markersize = 18, ls = (0, (5,5)), lw = 6, color = 'orange')
    sns.lineplot(x = adding_time, y = adding_pred, marker = 'o', markersize = 18, ls = (0,(5,5)), lw = 6, color = 'orange', alpha = .5)
    l.set_xticks(np.linspace(0, instants_array.max(), 6))
    l.set_xticklabels(np.array([1, 10, 20, 30, 40, 50]))
    
    plt.vlines(x = instants_array, ymin = 32, ymax = 40, color = 'grey', alpha = .3)
    plt.axvline(x = duration_cycle, color = 'red', ls = 'dotted', lw = 5) 
    plt.ylabel('')
    plt.legend(labels = ['$C_{100}$', '$\hat{C}_{100}$'], loc = 'lower left')

    plt_name = os.path.join('output', cell_name, 'meta', '{}_example.png'.format(analysis))
    plt.savefig(plt_name)
    
    plt.show()
    
    #plot_name
    
    
    
#plt.plot(instants_array[:len(y_hat)], y_hat)






