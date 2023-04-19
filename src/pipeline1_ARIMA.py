# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:45:36 2022

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
from src.utils.parameters import generate_param_grid
from src.utils.plotting import plot_prediction
from src.utils.cycle_selection import delete_anomalous_cycles
from pmdarima import arima, model_selection
import pmdarima as pm
import logging
from statsmodels.tsa.arima.model import ARIMA
from src.utils.cycle_selection import delete_anomalous_cycles
from statsmodels.stats.diagnostic import acorr_ljungbox


def main_ARIMA(n_instants = 50, var = None, cell_name = None):
    
    #LOGGER CONFIGURATION
    file_name = os.path.join('logs','ARIMA_{}_{}_T{}.log'.format(cell_name, var, n_instants))
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
    
       

    
    cfg_file = os.path.join('cfg', 'cfg.yaml')
    config = config_parser.parse_config(cfg_file)
    
    
    #ADHOC IN EACH FUNCTION
    an_type = 'ARIMA'
    approach = 'univariate'
    
    output_folder = config['output_folder']
    interpolators_folder = config['interpolators_folder']
    predictions_folder = config['predictions_folder']
    
    cell_path = config['data_path']
    cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
    with open(cell_file,'rb') as file:
        cell = pickle.load(file)
    file.close()
    

    c = config['c']

        
    logger.info('Variable to analyze: {}'.format(var))
    cycles = np.fromiter(cell[c].keys(),dtype = int)
    
    #Generate output path string
    output_path = os.path.join(output_folder, cell_name, approach, an_type, var)
    interp_path = os.path.join(interpolators_folder, cell_name)
    pred_path = os.path.join(output_path, predictions_folder)
    
    directories.validate_directory(interp_path)
    directories.validate_directory(output_path)
    directories.validate_directory(pred_path)
    
    
    # Filter valid cycles
    logger.info('Batch and cell: {}. Checking for anomalous cycles.'.format(cell_name))
    cell = delete_anomalous_cycles(cell, cycles)
    
    cycles = np.fromiter(cell[c].keys(),dtype = int)
    n_cycles = len(cycles)
    
    config = config_parser.convert_value(config, 'first_cycle', None, 0)
    config = config_parser.convert_value(config, 'last_cycle', None, n_cycles)
    config = config_parser.convert_value(config, 'step', None, 1)
    

    
    # Load interpolation functions
    
    logger.info('Loading interpolators.')
    
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
    
    nfolds = 5
    
    # Array of cycles to be predicted
    cycles_to_predict = np.arange(first_cycle, last_cycle,step)
    
    times_df = pd.DataFrame(columns = ['time'])
    
    logger.info('Number of instants: {}'.format(n_instants))
    
    #Fix instants
    fixed_instants = fix_instants(cell, n_instants)
    
    
    #Initialize df_instants
    first_cycle = cycles_to_predict[0]
    cycles = np.array(np.arange(first_cycle),dtype = 'str')
    
    #Initialize instants dataset
    logger.info('Initializing instants dataset.')
    #Initialize instants dataframe
    instants_array = np.arange(1,n_instants+1)
    cols = ['T{}'.format(i) for i in instants_array]
    #Loop over the explanatory cycles to fill the instants dataframe
    df_instants = pd.DataFrame(columns = cols)
    for cycle in cycles:    
        # estimate each cycles value in the fixed instants
        cycle_interp = interp_functions[str(cycle)](fixed_instants)
        
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
            
    
    
    #Generate parameter grid
    param_grid = generate_param_grid(an_type, first_cycle)
    parameter_grid = list(param_grid)
    ARIMA_models = dict()
    
    
    file_path = os.path.join(output_folder, cell_name, 'GS',var)
    file_name = os.path.join(file_path, 'ARIMA_T{}.pickle'.format(n_instants))
    
    if os.path.exists(file_name):
        logger.info('Metadata file exists. Reading.')
        with open(file_name, 'rb') as handle:
            ARIMA_models = pickle.load(handle)
        logger.info('Information loaded.')
    
    else:
        logger.info('Metadata file does not exist.')
        
        logger.info('The arange of cycles until the cycle 100 is used to search the best hyperparameters combination.')
        logger.info('Loop over the instants...')
    
        for n_instant, col in enumerate(df_instants.columns):
            logger.info('Time series [{}/{}]'.format(n_instant+1, n_instants))
            
            data = df_instants[col].to_numpy()
            

            if (np.isclose(data, data[0])).all():
                logger.info('Time series is constant.')
                p, q = None, None
                ARIMA_models[col] = (p, q)
                continue
            
            #check WHITE NOISE
            logger.info('Checking if the time series is just white noise.')
            logger.info('Ljung-Box test, lags = 10.')
            is_wn = (acorr_ljungbox(data, lags = 5)['lb_pvalue']>0.05).all()
            
            if is_wn:

                logger.info('The time series can be considered as white noise. Setting all parameters to 0.')
                p, q = 0, 0
    
            
            else:
            
                logger.info('We refuse the null hipotesis, so the time series is not white noise.')
                logger.info('Grid Search to select the hyperparameters in the time series.')
              
                
                mse_comb = np.array([])
                # Initialize models dictionary 
            
                empty = True
                mask = list()
                logger.info('Grid search over the parameter grid.')
                
                for comb in list(param_grid):
                    
                    
                    model_cv_errors = np.array([])
                    data_cv = data[:-nfolds-1]
                    
                    p, q = comb['p'], comb['param_1']
                    logger.info('p = {}, q = {}'.format(p,q))
                    for k_f in range(nfolds):
                        
                        #Initialize object               
                        d = arima.ndiffs(data, max_d = 3)
                        logger.info('Difference order: {}'.format(str(d)))
                
                        logger.info('Step [{}/{}] of CV process.'.format(k_f+1, nfolds))
                        y_true = data[-nfolds+k_f]
                        
                        if k_f == 0:
                            
                            ARIMA_cv_fit = None
                            logger.info('First fold. Initializing ARIMA model for the chosen parameters.')
                            ARIMA_cv = ARIMA(data_cv, order = (p, d, q))

                            try:
                                
                                logger.info('Trying to fit model...')
                                
                                ARIMA_cv_fit = ARIMA_cv.fit(method = 'innovations_mle')
                                
                                y_hat = ARIMA_cv_fit.forecast(1)
                                
                                mse = (y_true-y_hat)**2
                                
                            except ValueError as err:
                                logger.info('ValueError: {}. Updating d value.'.format(err))
                                d += 1
                                
                                logger.info('Trying with difference order d = {}'.format(d))
                                
                                try:
                                    ARIMA_cv = ARIMA(data_cv, order = (p, d, q))
                                    ARIMA_cv_fit = ARIMA_cv.fit(method = 'innovations_mle')
                                    y_hat = ARIMA_cv_fit.forecast(1)
                                
                                    mse = (y_true-y_hat)**2
                                    
                                except:
                                    logger.info('Did not work. Returning to previous difference order.')
                                    d -= 1
                                    continue
                                
                            except:
                                logger.info('Not possible to fit.')
                                mse = np.nan
                                
                        else:
                            
                            d = arima.ndiffs(data, max_d = 3)
                            logger.info('Diference order: {}'.format(str(d)))
                            
                            instant_to_append = data[-nfolds-1+k_f]
                            
                            #one step ahead prediction --> true value
                            data_cv = data[:-nfolds-1+k_f]
                            try:
                                if ARIMA_cv_fit == None:
                                    logger.info('Trying to fit new instance of ARIMA model.')
                                    ARIMA_cv = ARIMA(data_cv, order = (p, d, q))
                                    ARIMA_cv_fit = ARIMA_cv.fit(method = 'innovations_mle')
                                    
                                else:    
                                    logger.info('Appending one observation to the time series split and refitting.')
                                    ARIMA_cv_fit = ARIMA_cv_fit.append(np.array([instant_to_append]), refit = True)
                                    
                                y_hat = ARIMA_cv_fit.forecast(1)
                                
                                
                                mse = (y_true-y_hat)**2
                            
                                
                            except:
                                logger.info('Not possible to fit.')
                                mse = np.nan
                        
                        
                        model_cv_errors = np.append(model_cv_errors, mse)
    
                        
                    mean_fold_mse = np.nanmean(model_cv_errors)
                    
                    if mean_fold_mse == np.nan:
                        logger.info('Impossible to fit model for this parameter combination. Avoiding this combination.')
                        mean_fold_mse = 1e10
                    
                    
                    mse_comb = np.append(mse_comb, mean_fold_mse)
                    mask.append(empty)

                #Select best parameter set
                best_comb_index = np.argmin(mse_comb)
                
                logger.info('The best parameter set is the number {}.'.format(best_comb_index))
                
                best_comb = parameter_grid[best_comb_index]
                logger.info('The best parameter combination is: p = {}, d = {}, q = {}.'.format(best_comb['p'], d, best_comb['param_1']))
                
                
                p, q = best_comb['p'], best_comb['param_1']
            
            # Store ARIMA model hyperparameters in dict
            ARIMA_models[col] = (p, q)

        
        directories.validate_directory(file_path)
        logger.info('Grid Search finished. Store data.')
        with open(file_name, 'wb') as handle:
            pickle.dump(ARIMA_models, handle)
        logger.info('Stored.')
    
    
    
    
    logger.info('Loop over the cycles to predict...')

    for index,i in enumerate(cycles_to_predict):
        
        logger.info('Cycle to predict: {}'.format(i))
        # Get evaluation instants
        cycle_to_predict_data = get_evaluation_cycle(cell, i, fixed_instants)
        
        y_true = cycle_to_predict_data[var]
        selected_instants = cycle_to_predict_data['t']
        n_obs_evaluation = len(selected_instants)
        #n_obs_evaluation = len(fixed_instants)
                
        instants_array = np.arange(1,n_obs_evaluation+1)
        cols = ['T{}'.format(i) for i in instants_array]
        df_instants = pd.DataFrame(columns = cols)
        
        logger.info('Updating instants dataset...')
        
        cycles = np.arange(i)
        
        #Update instants dataset
        for cycle in cycles:
            # estimate each cycles value in the fixed instants
            cycle_interp = interp_functions[str(cycle)](selected_instants)
            #cycle_interp = interp_functions[str(cycle)](fixed_instants)
            
            
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
        t_pred_start = time.time()
        y_hat = np.array([])
    
       
        
        y_singular = np.array([])
        # Loop over the instants 
        for col in range(n_obs_evaluation):
            logger.info('Time series [{}/{}]'.format(col+1, n_obs_evaluation))
            
            col_name = 'T{}'.format(col+1)
            # Load only new instants data
            data = df_instants.loc[:, col_name].to_numpy()
            
            #Load ARIMA model
            p, q = ARIMA_models[col_name]
            d = arima.ndiffs(data, max_d = 3)
            #check if original time series was constant
            if ((p, q) == (None, None)) or ((p,q) == (0, 0)):
                logger.info('Original time series was considered constant.')
                logger.info('Prediction is done by computing the mean of the time series.')
                
                instant_pred = data.mean()
                y_hat = np.append(y_hat, instant_pred)
                
                continue
            
            logger.info('The order value p is not 0.')
            ARIMA_model = ARIMA(data, order=(p, d, q))
            
            fit = False
            fit_method = None
            
            
            while not fit:
                options = {'minimize_kwargs':{'options':{'maxiter': 100}}}
                if not fit_method:
                    logger.info('First iteration. Trying to estimate the parameters via "innovations_mle" procedure.')
                    fit_method = 'innovations_mle'
                
                try:
                    logger.info('Trying to fit ARIMA({}, {}, {}) model.'.format(p, d, q))
                    
                    ARIMA_fit = ARIMA_model.fit(method = fit_method, method_kwargs = options)
                    
                    #verify fit
                    logger.info('Model fitted. Checking the optimization procedure.')
                    if fit_method == 'innovations_mle':
                        if d!= 0:
                            wf = ARIMA_fit.fit_details['minimize_results']['status']
                        else:
                            if ARIMA_fit.fit_details['converged']:
                                wf = 0
                            else:
                                wf = 3
                    else:
                        wf = ARIMA_fit.fit_details.mle_retvals['warnflag']
                    
                    #chequeamos esto
                    if fit_method == 'innovations_mle' and wf != 0:
                        logger.info('The optimization process did not converged, the model may be unstable.')
                        logger.info('Trying to fit using another estimation method.')
                        
                        d_kpss = arima.ndiffs(data, max_d = 3)
                        ARIMA_model_ss = ARIMA(data, order = (p, d_kpss, q))
                        
                        ARIMA_fit_ss = ARIMA_model_ss.fit(method = 'statespace')
                        
                        wf = ARIMA_fit_ss.fit_details.mle_retvals['warnflag']
                        
                        logger.info('Checking the results of the optimization procedure.')
                        if wf == 0:
                            logger.info('The optimization procedure converged. We keep this model.')
                            ARIMA_fit = ARIMA_fit_ss
                            fit = True
                            
                        else:
                            logger.info('The optimization procedure did not converged. We compare the fitted values and we keep the best model.')
                            logger.info('Comparing model residuals.')
                            
                            d_first = max(d, d_kpss)
                            r_innovations = ARIMA_fit.resid[d_first:]
                            r_ss = ARIMA_fit_ss.resid[d_first:]
                            
                            r_innovations_cusum = r_innovations.T@r_innovations
                            r_ss_cusum = r_ss.T@r_ss
                            
                            if r_ss_cusum<r_innovations_cusum:
                                logger.info('The statespace estimation procedure is better than the innovations_mle.')
                                
                                ARIMA_fit = ARIMA_fit_ss
                                fit = True
                                
                            else:
                                logger.info('The innovations_mle procedure is better than the statespace method.')
                                fit = True
                    else:
                        logger.info('The estimation procedure has converged.')
                        fit = True
                except:
                    
                    if fit_method == 'statespace':
                        logger.info('FATAL ERROR.')
                        wf = 4
                        instant_pred = 0
                        break
                        
                    logger.info('Error. Augmenting the difference order.')
                    
                    d += 1
                    
                    logger.info('Difference order d = {}.'.format(d))
                    ARIMA_model = ARIMA(data, order=(p, d, q))
                    
                    if d>3:
                        logger.info('Difference order greater than 3. Not possible to make forecasts.')
                        logger.info('Last chance. Trying the state-space estimation method.')
                        
                        #return to original difference order
                        d = arima.ndiffs(data, max_d = 3)
                        ARIMA_model = ARIMA(data, order=(p, d, q))
                        fit_method = 'statespace'
                    
            if fit:
                instant_pred = ARIMA_fit.forecast(1)
            
            logger.info('Predicted value: {}'.format(instant_pred))
            y_hat = np.append(y_hat,instant_pred)
    
                        
                
        t_pred = time.time()-t_pred_start
    
        # Store prediction & results
        logger.info('Storing prediction data.')
        to_store = pd.DataFrame({'y':y_true,'y_hat': y_hat}, index = selected_instants)
        to_store = to_store.dropna()
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





