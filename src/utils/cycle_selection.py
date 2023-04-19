"""
Created on Wed Apr  6 09:51:27 2022

@author: aarregui
"""

import numpy as np


def delete_anomalous_cycles(cell, cycles, c = 'cycles'):
    
    '''
    Remove the cycles that are anomalous based on their duration.

    Parameters
    ----------
    cell : dict
        Cell data.
        
    cycles : np.array
        Cycles array
    
    c : str
        Dictionary key name.
        

    Returns
    -------
    cell : dict
        Dictionary containing all the interpolators.

    '''    
    
    
    duration_array = np.array([])
    nobs_array = np.array([])
    
    for cycle in cycles:
        
        t = cell[c][str(cycle)]['t']
        
        t_max = t.max()
        n_obs = len(t)
        
        duration_array = np.append(duration_array, t_max)
        nobs_array = np.append(nobs_array, n_obs)

    dur_mean = duration_array.mean()
    dur_std = duration_array.std()
    
    
    k = 5
    
    threshold_min = np.max([dur_mean - dur_std*k, 10]) 
    threshold_max = dur_mean +  dur_std*k
    
    n_outliers_max = (threshold_max<duration_array).sum()
    n_outliers_min = (threshold_min>duration_array).sum()
        

    cycles_to_delete = np.array([], dtype = int)
    
    
    if n_outliers_max>0:
        
        index = threshold_max<duration_array
        cycles_to_delete = np.append(cycles_to_delete, cycles[index])
        
        
    if n_outliers_min>0:
        
        index = threshold_min>duration_array
        cycles_to_delete = np.append(cycles_to_delete, cycles[index])
        
    
    for cycle in cycles_to_delete:
        del cell[c][str(cycle)]
    
    
    
    
    old_cycles = np.fromiter(cell[c].keys(), dtype = int)
    n_cycles = len(cell[c].keys())
    new_cycles = np.arange(n_cycles)
    
    cycles_dict = dict()
    for new_cycle, old_cycle in zip(new_cycles, old_cycles):
        
        cycles_dict[str(new_cycle)] = cell[c][str(old_cycle)]
    
    
    cell[c] = cycles_dict
    
    
    return cell

















