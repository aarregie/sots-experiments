# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:32:43 2022

@author: aarregui
"""

#%%
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)
from src.utils import config_parser, directories
from src.utils.cycle_selection import delete_anomalous_cycles
from scipy.interpolate import interp1d


cfg_file = os.path.join('cfg', 'cfg.yaml')
config = config_parser.parse_config(cfg_file)

cell_path = config['data_path']


cells = os.listdir(cell_path)

cells = [cell.replace('.pickle', '') for cell in cells]

variables = ['I', 'Qc', 'Qd', 'T', 'V']

c = config['c']
interpolators_folder = config['interpolators_folder']

for cell_name in cells:
    cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
    with open(cell_file,'rb') as file:
        cell = pickle.load(file)
    
    interpolator_path = os.path.join(interpolators_folder, cell_name)
    
    directories.validate_directory(interpolator_path)

    cycles = np.fromiter(cell[c].keys(),dtype = int)
    
    cell = delete_anomalous_cycles(cell, cycles)
    
    cycles = np.fromiter(cell[c].keys(),dtype = int)
    n_cycles = len(cycles)

    
    cycle_keys = cell[c].keys()
    
    for variable in variables:    
        
        print(variable)
        interp_file_name = os.path.join(os.path.join(interpolator_path,'interp_var{}.pkl'.format(variable)))
        
    
        interp_functions = dict()
    
        for cycle in cycle_keys:
            print(cycle)
            X = cell[c][cycle]['t']
            y = cell[c][cycle][variable]
            
            y_interp = interp1d(X, y, fill_value = "extrapolate")
            interp_functions[cycle] = y_interp
        
        #Store functions
        with open(interp_file_name, 'wb') as f:
            pickle.dump(interp_functions, f)
        f.close()
           
        
            







