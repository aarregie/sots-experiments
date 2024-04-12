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

cfg_file = os.path.join('cfg', 'cfg.yaml')
config = config_parser.parse_config(cfg_file)
c = config['c']

cell_path = os.path.join('data', 'cell')

#list data

cells = os.listdir(cell_path)

var = 'T'

cell_name = config['cell']


cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
with open(cell_file,'rb') as file:
    cell = pickle.load(file)
file.close()


#%%

cycles_up = ['100', '120', '140'] 
cycles_down = ['1', '490', '980']

linestyles_up = [('solid', 'solid'), ('dotted', 'dotted'), ('dashed', 'dashed')]
linestyles_down = [('loosely dashed', (0, (10, 2))), 
                   ('loosely dashdotted', (0, (8, 3, 3, 3))),
                   ('loosely dashdotdotted', (0, (1, 5, 1, 5, 1, 5)))]

colors_up = ['darkorange', 'limegreen', 'royalblue']
colors_down = ['darkgreen', 'cadetblue', 'red']

fig, axs = plt.subplots(2,1, sharex=True, figsize=(30, 15))

file_name = os.path.join('output', 'general_results', 'boxplots', 'cell_TS.png')

for index, cycle in enumerate(cycles_up):
    
    cycle_label = 'Cycle {}'.format(cycle)
    color_up = colors_up[index]
    linestyle = linestyles_up[index][1]
    cell_data = cell[c][cycle][var]
    cell_time = cell[c][cycle]['t']
    
    axs[0].plot(cell_time, cell_data, c = color_up, lw = 5, ls = linestyle, label = cycle_label)

axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 35)
axs[0].yaxis.set_tick_params(labelsize=40)


for index, cycle in enumerate(cycles_down):
    
    cycle_label = 'Cycle {}'.format(cycle)
    color_down = colors_down[index]
    linestyle = linestyles_down[index][1]
    cell_data = cell[c][cycle][var]
    cell_time = cell[c][cycle]['t']
    
    axs[1].plot(cell_time, cell_data, c = color_down, lw = 5, ls = linestyle, label = cycle_label)


axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 35)
axs[1].yaxis.set_tick_params(labelsize=40)
axs[1].xaxis.set_tick_params(labelsize=40)
axs[1].set_xlabel('Time (min)', fontsize = 45)
fig.supylabel('Temperature', fontsize = 50)
fig.tight_layout()
plt.savefig(file_name)
plt.show()    
    







