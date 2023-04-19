# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:24:37 2022

@author: aarregui
"""
#%%
## EXPLORE

import sys
import os
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30, 20)
from src.utils import config_parser, directories
from src.utils.instants import *
from src.utils.parameters import generate_param_grid
from src.utils.plotting import plot_prediction
from src.utils.cycle_selection import delete_anomalous_cycles
import pmdarima as pm
import logging
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator


#%%

cfg_file = os.path.join('cfg', 'cfg.yaml')
config = config_parser.parse_config(cfg_file)


output_folder = config['output_folder']
interpolators_folder = config['interpolators_folder']
predictions_folder = config['predictions_folder']

cell_path = config['data_path']
cell_name = config['cell']



# Abrimos un ejemplo.
# El cell que se analizará es el b2c2 --> batch 2, cell 2

cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
with open(cell_file,'rb') as file:
    cell = pickle.load(file)
file.close()

s = config['s']
c = config['c']

cqd = config['var']


n_instants = config['n_instants']

#Generate output path string
output_path = os.path.join(output_folder, cell_name)
interp_path = os.path.join(interpolators_folder, cell_name)


directories.validate_directory(interp_path)
directories.validate_directory(output_path)

#carpeta de metadatos para cada cell
meta_path = os.path.join(output_path, 'meta')
directories.validate_directory(meta_path)


cycles = np.fromiter(cell[c].keys(),dtype = int)
n_cycles = len(cycles)

cell = delete_anomalous_cycles(cell, cycles)
    
cycles = np.fromiter(cell[c].keys(),dtype = int)
n_cycles = len(cycles)-1

step = config['step']

config = config_parser.convert_value(config, 'first_cycle', None, 0)
config = config_parser.convert_value(config, 'last_cycle', None, n_cycles)
config = config_parser.convert_value(config, 'step', None, 1)


#%%

#Chequeamos qué variables podemos usar

example = cell[c]['100']

example_t = example['t']
variables = ['I', 'Qc', 'Qd', 'T', 'V']

n_obs = len(example_t)

for var in variables:
    example_var = example[var]
    
    n_var = len(example_var)
    
    if n_var == n_obs:
        print('{} OK'.format(var))
    
    else:
        print('{} NOT OK.'.format(var))
        
        


#%%

var = 'T'
var_human = 'Temperature'

cycles_arange = np.arange(0, n_cycles, step).astype('str')

x = np.array([])
y = np.array([])
z = np.array([])

for cycle in cycles_arange:
    
    ts= cell[c][cycle]['t']
    ys = np.array(np.repeat(int(cycle), len(ts)), dtype = int)
    zs = cell[c][cycle][var]
    
    #cogemos n veces menos datos
    n_obs = len(ts)
    n = 4
    n_points = n_obs//n
    
    index = np.linspace(0, n_obs-1, n_points, dtype = int)
    
    ts = ts[index]
    ys = ys[index]
    zs = zs[index]
    
    x = np.append(x, ts)
    y = np.append(y, ys)
    z = np.append(z, zs)
    
    
y = np.array(y, dtype = int)





#%%
# print selected cycles

#plot line
last_plot = y[-1]
mid_plot = y[-1]//2
selected_cycles = ['0', str(mid_plot), str(last_plot)]



for selected_cycle in selected_cycles:
    
    t_sc = cell[c][selected_cycle]['t']
    
    d = {'time': t_sc}
    
    for cqd in variables:
        y_sc = cell[c][selected_cycle][cqd]
    
        d[cqd] = y_sc
        
        if cqd == var:
            var_y = y_sc
            plt.plot(t_sc, y_sc, label = selected_cycle)
        
    df = pd.DataFrame(d)
    
    csv_file_name = os.path.join(meta_path, 'cycle_{}.csv'.format(selected_cycle))
    df.to_csv(csv_file_name, index = False)
    
    
plt.legend(fontsize=20)
plt.xlabel('Time')
plt.ylabel(var)
plt.grid(False)

#x label
plt.xlabel('Time', labelpad = 20, fontsize = 20)
xticks = np.array([0, 10, 20, 30, 40, 50])
plt.xticks(xticks, labels = xticks, size = 15)
plt.xlabel('Time', labelpad = 20, fontsize = 20)

#y label
y_min, y_max = np.round(var_y.min()), np.ceil(var_y.max())
y_ticks = np.round(np.linspace(y_min, y_max, 4),1)
plt.yticks(y_ticks,labels = y_ticks, size = 15)
plt.ylabel(var, labelpad = 20, fontsize = 20)


plt.show()



#%%
# print consecutive cycles
#plot line
last_plot = y[-1]
mid_plot = y[-1]//2
selected_cycles = ['100', '120', '140']

for selected_cycle in selected_cycles:
    
    t_sc = cell[c][selected_cycle]['t']
    
    d = {'time': t_sc}
    
    for cqd in variables:
        y_sc = cell[c][selected_cycle][cqd]
    
        d[cqd] = y_sc
        
        if cqd == var:
            plt.plot(t_sc, y_sc, label = selected_cycle)
            var_y = y_sc
    df = pd.DataFrame(d)
    
    csv_file_name = os.path.join(meta_path, 'consecutive_cycle_{}.csv'.format(selected_cycle))
    df.to_csv(csv_file_name, index = False)
    
    
plt.legend(fontsize=20)
plt.xlabel('Time')
plt.ylabel(var)
plt.grid(False)

#x label
plt.xlabel('Time', labelpad = 20, fontsize = 20)
xticks = np.array([0, 10, 20, 30, 40, 50])
plt.xticks(xticks, labels = xticks, size = 15)
plt.xlabel('Time', labelpad = 20, fontsize = 20)

#y label
y_min, y_max = np.round(var_y.min()), np.ceil(var_y.max())
y_ticks = np.round(np.linspace(y_min, y_max, 4),1)
plt.yticks(y_ticks,labels = y_ticks, size = 15)
plt.ylabel(var, labelpad = 20, fontsize = 20)


plt.show()





#%%

plt.rcParams.update({'font.size': 35})
triang = mtri.Triangulation(x, y)


fig = plt.figure(frameon=True)


ax = fig.add_subplot(111,projection='3d')

surf = ax.plot_trisurf(triang, z, cmap='viridis_r',edgecolor = 'none')

ax.set_xlabel('Time', labelpad = 40, fontsize = 30)
ax.set_ylabel('Cycles', labelpad = 50, fontsize = 30)

#x range
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50]))
ax.set_xticklabels(labels = np.array([0, 10, 20, 30, 40, 50]) ,size = 30)


ax.set_facecolor('white')

ax.tick_params(pad = 25)


#z range
z_min, z_max = np.round(z.min()), np.ceil(z.max())
z_ticks = np.round(np.linspace(z_min, z_max, 4),1)
ax.set_zticks(z_ticks)
ax.set_zticklabels(labels = z_ticks, size = 30)



#y range
y_ticks = np.array(np.linspace(0,last_plot+1,5), dtype = int)
y_ticks = np.arange(0,last_plot+1,200)
ax.set_yticks(y_ticks)
ax.set_yticklabels(labels = y_ticks, size = 30)
ax.yaxis.set_ticks_position('none') 



ax.view_init(elev=40, azim=-60)
ax.grid(False)
cbar = fig.colorbar(surf, shrink = 0.5)
cbar.ax.tick_params(labelsize=30)


plot_name = os.path.join(meta_path, '3d_{}.png'.format(var))

plt.savefig(plot_name)

plt.show()




