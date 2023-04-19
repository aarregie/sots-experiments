# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:27:50 2022

@author: aarregui
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)
from src.utils import config_parser, directories
from src.utils.instants import *
from src.utils.cycle_selection import delete_anomalous_cycles
import logging
from src.utils.parameters import generate_param_grid
from src.utils.plotting import plot_prediction
from src.utils.CV import GridSearchCV_TimeSeries
from src.utils.auxiliar import get_model
from src.utils.plotting import plot_VAR_params, plot_prediction


def main_LRVAR(n_instants = 50, var = None, cell_name = None):
    #LOGGER CONFIGURATION
    file_name = os.path.join('logs','LRVAR_{}_{}_T{}.log'.format(cell_name, var, n_instants))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(file_name, mode = 'w')
    handler.setFormatter(formatter)
    show_output_screen = False
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        print('Clearing handlers.')
        logger.handlers.clear()
    
    
    #add handlers
    logger.addHandler(handler)
    if show_output_screen:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
    
    
    
    logger.info('Starting cycle prediction experiments. Multivariate pipeline.')
    
    cfg_file = os.path.join('cfg', 'cfg.yaml')
    config = config_parser.parse_config(cfg_file)
    
    #ADHOC IN EACH FUNCTION
    an_type = 'LRVAR'
    approach = 'multivariate'
    
    
    logger.info('Configurating analysis. Analysis type: {}'.format(an_type))
    
    output_folder = config['output_folder']
    interpolators_folder = config['interpolators_folder']
    predictions_folder = config['predictions_folder']
    parameters_folder = config['parameters_folder']
    
    cell_path = config['data_path']
    
    cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
    with open(cell_file,'rb') as file:
        cell = pickle.load(file)
    file.close()
    
    c = config['c']

        
    logger.info('Variable to analyze: {}'.format(var))
        
    #Generate output path string
    output_path = os.path.join(output_folder, cell_name, approach, an_type, var)
    interp_path = os.path.join(interpolators_folder, cell_name)
    pred_path = os.path.join(output_path, predictions_folder)
    params_path = os.path.join(output_path, parameters_folder)
    logger.debug('Creating necessary folders & paths.')
    directories.validate_directory(interp_path)
    directories.validate_directory(output_path)
    directories.validate_directory(pred_path)
    directories.validate_directory(params_path)
    
    
    cycles = np.fromiter(cell[c].keys(),dtype = int)
    
    # Filter valid cycles
    logger.info('Batch and cell: {}. Checking for anomalous cycles.'.format(cell_name))
    cell = delete_anomalous_cycles(cell, cycles)
    
    cycles = np.fromiter(cell[c].keys(),dtype = int)
    n_cycles = len(cycles)
    
    config = config_parser.convert_value(config, 'first_cycle', None, 0)
    config = config_parser.convert_value(config, 'last_cycle', None, n_cycles)
    config = config_parser.convert_value(config, 'step', None, 1)
    
    
    logger.info('Loading interpolators.')
    
    # Load interpolation functions
    
    interp_file_name = os.path.join(os.path.join(interp_path,'interp_var{}.pkl'.format(var)))
    
    with open(interp_file_name, 'rb') as f:
        interp_functions = pickle.load(f)
    f.close()
    
       
    #################
    ###GRID SEARCH###
    #################
        
    
    first_cycle = config['first_cycle']
    last_cycle = config['last_cycle']
    step = config['step']
    
    pred_step = 1
    
    times_df = pd.DataFrame(columns = ['time'])
    

    logger.info('Number of instants: {}'.format(n_instants))
    
    
    
    #Initialize instants dataframe
    instants_array = np.arange(1,n_instants+1)
    
    cols = ['T{}'.format(i) for i in instants_array]
    df_instants = pd.DataFrame(columns = cols)
    
    model = get_model(an_type)
    
    
    # Array of cycles to be predicted
    cycles_to_predict = np.arange(first_cycle, last_cycle, step)
            
    #Fix instants
    fixed_instants = fix_instants(cell, n_instants)
    
    
    #Initialize df_instants
    first_cycle = cycles_to_predict[0]
    cycles = np.array(np.arange(first_cycle),dtype = 'str')
    
    #Initialize instants dataset
    logger.info('Initializing dataset.')
    #Initialize instants dataframe
    instants_array = np.arange(1,n_instants+1)
    cols = ['T{}'.format(i) for i in instants_array]
    #Loop over the explanatory cycles to fill the instants dataframe
    df_instants = pd.DataFrame(columns = cols)
    for cycle in cycles:    
        # estimate each cycles value in the fixed instants
        #cycle_interp = interp_functions[str(cycle)](fixed_instants)
        cycle_interp = interp_functions[str(cycle)](selected_instants)
        
        ys_true = cell[c][str(cycle)][var]
            
        y_min, y_max, y_last = ys_true.min(), ys_true.max(), ys_true[-1]
        
        for index in range(len(cycle_interp)):
                  
            y_hat = cycle_interp[index]
            
            if (y_hat<y_min) or (y_hat>y_max):
                
                cycle_interp[index] = y_last
        
        df_append = pd.DataFrame([cycle_interp], columns = cols, index = [cycle])
    
        if df_instants.empty:
            df_instants = df_append.copy()
        else:
            df_instants = pd.concat([df_instants, df_append])
            
    
    
    #Parameter grid
    param_grid = generate_param_grid(an_type, first_cycle)
    
    
    logger.info('Grid Search to select the hyperparameters in the fixed instants.')
    logger.info('The arange of cycles until the cycle {} is used to search the best hyperparameters combination.'.format(first_cycle))
    Y_matrix = df_instants.to_numpy().T
    
    #LEARN MODELS
    logger.info('Initializing CV Grid search.')
    cv_result = GridSearchCV_TimeSeries(Y_matrix, an_type, param_grid)
    
    param_1, p = cv_result['best_model']['param_1'], cv_result['best_model']['p']
        
    
    
    logger.info('Grid Search finished.')
    
    
    #################
    ###FORECASTING###
    #################
    
        
    times_df = pd.DataFrame(columns = ['time'])
    
    
    logger.info('Loop over the cycles to predict...')
    for index,i in enumerate(cycles_to_predict):
        
        t_pred_start = time.time()
        logger.info('Cycle to predict: {}'.format(i))
        # Get evaluation instants
        cycle_to_predict_data = get_evaluation_cycle(cell, i, fixed_instants)
        
        y_true = cycle_to_predict_data[var]
        selected_instants = cycle_to_predict_data['t']
        n_obs_evaluation = len(fixed_instants)
        
        instants_array = np.arange(1,n_obs_evaluation+1)
        cols = ['T{}'.format(i) for i in instants_array]
        df_instants = pd.DataFrame(columns = cols)
        
        logger.info('Updating instants dataset...')
        
        cycles = np.arange(i)
        #Update instants dataset (add observations) and update models.
        for cycle in cycles:
            # estimate each cycles value in the fixed instants
            #cycle_interp = interp_functions[str(cycle)](fixed_instants)
            cycle_interp = interp_functions[str(cycle)](selected_instants)
            
            ys_true = cell[c][str(cycle)][var]
        
            y_min, y_max, y_last = ys_true.min(), ys_true.max(), ys_true[-1]
            
            for index in range(len(cycle_interp)):
                      
                y_hat = cycle_interp[index]
                
                if (y_hat<y_min) or (y_hat>y_max):
                    
                    cycle_interp[index] = y_last
            
            df_append = pd.DataFrame([cycle_interp], columns = cols, index = [cycle])
        
            if df_instants.empty:
                df_instants = df_append.copy()
            else:
                df_instants = pd.concat([df_instants, df_append])
        
        logger.info('Instants dataset updated.')
        logger.info('Updating models and making predictions...')
        
        # Training dataset
        Y_matrix = df_instants.to_numpy().T
        #train model
        results = model(Y_matrix, p, param_1, pred_step)
        #prediction
        y_hat, comb_A  = results[0][:,-1].ravel(), results[1]
    
                
        t_pred = time.time()-t_pred_start

        # Store prediction & results
        logger.info('Storing prediction data.')
        to_store = pd.DataFrame({'y':y_true,'y_hat': y_hat}, index = selected_instants)
        to_store.index.name = 'time'
        file_name = os.path.join(pred_path,'pred_{}_C{}_T{}.csv'.format(var,i,n_instants))
        
        to_store.to_csv(file_name)
        
       
        plt_name = os.path.join(pred_path, 'pred_{}_C{}_T{}.png'.format(var,i,n_instants))
        logger.info('Storing prediction plot.')
        plot_title = 'Cycle {} forecast VS observed'.format(i)
        plot = plot_prediction(selected_instants, y_true, y_hat, title = plot_title, 
                               y_label = 'Observed', y_hat_label = 'Forecast',
                               x_axis_label = 't', y_axis_label = var, plt_name = plt_name)
        
        

        logger.info('Storing execution time.')
        times_df.loc[i, 'time'] = t_pred
    
    
        time_file = os.path.join(output_path, 'time_T{}.csv'.format(n_instants))
        times_df.to_csv(time_file)
    
    
    logger.info('End.')
    logging.shutdown()
    
    
    
