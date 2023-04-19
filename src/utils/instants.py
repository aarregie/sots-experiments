"""
Created on Wed Apr  6 09:51:27 2022

@author: aarregui
"""
import numpy as np


def get_cycles_len(data):
    '''
    This function returns the highest value of the cycles time series instants.

    Parameters
    ----------
    data : dict
        Cell data.

    Returns
    -------
    cycles_len : float
        Highest value of the instant.

    '''
    
    #Initialize array
    cycles_len = np.array([])
    cycles = np.fromiter(data['cycles'].keys(),dtype = int)
    n_cycles = len(data['cycles'])
    
    #Loop over arrays
    for cycle in cycles:
        #Get cycle duration
        cycle_len = max(data['cycles'][str(cycle)]['t'])
        #Append to duration array
        cycles_len = np.append(cycles_len, cycle_len)
        
    return cycles_len


def get_slowest_cycle_index(data):
    '''
    This function returns the index of the slowest cycle in the cell data.

    Parameters
    ----------
    data : dict
        Cell data.

    Returns
    -------
    slowest_cycle_index : int
        Index of the slowest cycle.

    '''
    
    # Get cycles duration array
    cycles_len = get_cycles_len(data)
    
    # Get slowest cycle index
    slowest_cycle_index = np.argmax(cycles_len)
    
    
    return slowest_cycle_index

def get_cycle(data, cycle_index):
    '''
    
    This function returns the cycle data in the given index.

    Parameters
    ----------
    data : dict
        Cell data.
    
    cycle_index : int
        Cycle index.
    Returns
    -------
    Cycle : dict
        Cycle data.

    '''
    
    # Get slowest cycle data
    cycle = data['cycles'][str(cycle_index)]
    
    return cycle



def get_slowest_cycle(data):
    '''
    
    This function returns the slowest cycle data.

    Parameters
    ----------
    data : dict
        Cell data.

    Returns
    -------
    slowest_cycle : dict
        Slowest cycle data.

    '''
    #Initialize result
    slowest_cycle = dict()
    
    # Get slowest cycle index
    slowest_cycle_index = get_slowest_cycle_index(data)
    
    # Get slowest cycle data
    slowest_cycle = get_cycle(data, slowest_cycle_index)
    
    return slowest_cycle





def get_cycle_time_data(cycle):
    '''
    Get number of values and duration of the TS.
    

    Parameters
    ----------
    cycle : Dict
        Dictionary containing the data of the TS.

    Returns
    -------
    result : tuple
        (number of observations, duration of the TS)

    '''
    # Initialize result
    result = tuple()
    # Get number of observations
    cycle_n_obs = len(cycle['t'])
    
    #Get cycle instants array
    cycle_t = cycle['t']
    
    # Get result
    result = cycle_n_obs, cycle_t
    
    return result



def fix_instants(data, n_instants = 50):
    '''
    Fix instants in the slowest cycle of the whole dataset.

    Parameters
    ----------
    data : dict
        Dictionary containing the Sequence of Time Series.
    n_instants : int, optional
        Number of instants to select in the slowest cycle. The default is 50.

    Returns
    -------
    instants : np.array
        Numpy ndarray containing the instants values.

    '''
    
    #Initialize result
    instants = np.array([])
    #Get slowest cycle data
    slowest_cycle = get_slowest_cycle(data)
    
    #Get slowest cycle instants
    slowest_cycle_t = slowest_cycle['t']
    #Get number of observations
    slowest_n_obs = len(slowest_cycle_t)
    
    #Select index of the fixed instants
    selected_instants_idx = np.linspace(0, slowest_n_obs-1, n_instants, endpoint= False, dtype ='int')
    
    # Fix instants
    instants = slowest_cycle_t[selected_instants_idx]
    
    return instants




def get_evaluation_instants_mask(data, cycle_to_predict, fixed_instants):
    '''
    Function to get the instants of the cycle to predict that are closest to the
    ones fixed.

    Parameters
    ----------
    cycle_to_predict : int
        Cycle to predict.
    data : dict
        Data containing the cycles time series.
    fixed_instants : np.array
        np.ndarray containing the fixed instants.

    Returns
    -------
    evaluation_instants : np.array
        np.ndarray containing the instants to use in the evaluation of the prediction.

    '''
    

    

    # Get prediction cycle data
    cycle_to_predict_data = get_cycle(data, cycle_to_predict)
    n_obs, cycle_to_predict_t = get_cycle_time_data(cycle_to_predict_data)
    
    #Initialize mask
    cmask = np.zeros(n_obs)
    max_cycle_t = cycle_to_predict_t.max()
    
    
    for instant in fixed_instants:

        if instant <= max_cycle_t and instant != 0:
            diff_instants = abs(cycle_to_predict_t - instant)
            mask_idx = np.argmin(diff_instants)
            
            cmask[mask_idx] = 1
        if instant == 0:
            
            cmask[0] = 1
            
    cmask_bool = cmask == 1
    
    
    return cmask_bool





def get_evaluation_cycle(data, cycle_to_predict, fixed_instants):
    '''
    Function to get the instants of the cycle to predict that are closest to the
    ones fixed.

    Parameters
    ----------
    cycle_to_predict : int
        Cycle to predict.
    data : dict
        Data containing the cycles time series.
    fixed_instants : np.array
        np.ndarray containing the fixed instants.

    Returns
    -------
    evaluation_instants : np.array
        np.ndarray containing the instants to use in the evaluation of the prediction.

    '''

    #Initialize evaluation_instants
    
    cmask_bool = get_evaluation_instants_mask(data, cycle_to_predict, fixed_instants)
    cycle_to_predict = get_cycle(data, cycle_to_predict)
    
    cycle_keys = ['I', 'Qc', 'Qd', 'T', 'V', 't']
    
    for key in cycle_keys:
        cycle_to_predict[key] = cycle_to_predict[key][cmask_bool]
    
    
    return cycle_to_predict
