# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:09:20 2022

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
import logging
    
from src.utils.cycle_selection import delete_anomalous_cycles
from src.utils.instants import *
import torch
import gpytorch
from src.utils.MTGP_model_variational import MultitaskGPModel
from src.utils.plotting import plot_prediction

from sklearn.preprocessing import StandardScaler



def main_MTGP(n_instants = 50, var = None, cell_name = None):
    #LOGGER CONFIGURATION
    file_name = os.path.join('logs','MTGP_{}_{}_T{}.log'.format(cell_name, var, n_instants))
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
    an_type = 'MTGP'
    approach = 'multivariate'
    
    output_folder = config['output_folder']
    
    interpolators_folder = config['interpolators_folder']
    predictions_folder = config['predictions_folder']
    parameters_folder = config['parameters_folder']
    
    
    logger.info('Configurating analysis. Analysis type: {}'.format(an_type))
    
    cell_path = config['data_path']
    
    #Open data file
    cell_file = os.path.join(cell_path,'{}.pickle'.format(cell_name))
    with open(cell_file,'rb') as file:
        cell = pickle.load(file)
    file.close()
    
    #names of the data dictionary keys
    c = config['c']
    
        
    logger.info('Variable to analyze: {}'.format(var))
        
    #Generate output path
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
    
    
    
    training_iterations = 1000
    interp_file_name = os.path.join(os.path.join(interp_path,'interp_var{}.pkl'.format(var)))
    
    
    logger.info('Loading interpolators.')
    
    # Load interpolation functions    
    interp_file_name = os.path.join(os.path.join(interp_path,'interp_var{}.pkl'.format(var)))
    
    with open(interp_file_name, 'rb') as f:
        interp_functions = pickle.load(f)
    f.close()
    
        
    first_cycle = config['first_cycle']
    last_cycle = config['last_cycle']
    step = config['step']
    

    logger.info('Number of instants: {}'.format(n_instants))
    #Fix instants
    fixed_instants = fix_instants(cell, n_instants)
    
    cycles_to_predict = np.arange(first_cycle, last_cycle, step)
    
    logger.info('Loop over the cycles to predict...')
    # Lectura de los interpoladores
    
    times_df = pd.DataFrame(columns = ['time'])

    
    latents_num = 5
    
    
    
    for index, i in enumerate(cycles_to_predict):
        
        normalizer = StandardScaler()
        t_pred = time.time()
        cycle_to_predict = str(i)
        logger.info('Cycle to predict: {}'.format(cycle_to_predict))
    
        
        
        cycle_to_predict_data = get_evaluation_cycle(cell, i, fixed_instants)
        
        y_true = cycle_to_predict_data[var]
        selected_instants = cycle_to_predict_data['t']
        #n_obs_evaluation = len(fixed_instants)
        n_obs_evaluation = len(selected_instants)
        
        instants_array = np.arange(1,n_obs_evaluation+1)
        
        cols = ['T{}'.format(i) for i in instants_array]
        df_instants = pd.DataFrame(columns = cols)
        
        logger.info('Updating instants dataset...')
        
        cycles = np.arange(i)
        last_cycle = len(df_instants)
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
                
    
        logger.info('Interpolators dictionary deleted. Dataset composed.')
        logger.info('Building model: Multi-Task Gaussian Process.')
        
        
        train_x = torch.from_numpy(df_instants.index.to_numpy()).float()
        y_matrix = df_instants.copy().to_numpy()
        
        # Normalize data
        y_matrix = normalizer.fit_transform(y_matrix)
        train_y = torch.from_numpy(y_matrix).float()
        
        tasks_num = n_obs_evaluation
        n_inducing_points = (len(train_x)//latents_num) * latents_num
        inducing_points = train_x[-n_inducing_points:].numpy()
        
        model = MultitaskGPModel(tasks_num, latents_num, inducing_points)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks_num, 
                                                                      noise_constraint=gpytorch.constraints.GreaterThan(1e-5),
                                                                      noise_prior=gpytorch.priors.NormalPrior(0, 1))
        

        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)  # Includes GaussianLikelihood parameters
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        

        logger.info('Starting optimization...')
        for k in range(training_iterations):
            # Within each iteration, we will go over each minibatch of data
            
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)

            loss.backward()
            logger.info('Iteration [{}/{}] - Loss: {}'.format(k+1, training_iterations, loss.item()))
            optimizer.step()
        
        logger.info('Fin MLL algorithm.')
        
        model.eval()
        likelihood.eval()
        
        # Make predictions
        logger.info('Make predictions.')
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.arange(int(cycle_to_predict),int(cycle_to_predict)+1).float()
            predictions = likelihood(model(test_x))
            y_hat_normalized = predictions.mean.numpy()
            y_hat = normalizer.inverse_transform(y_hat_normalized).ravel()
            #lower, upper = predictions.confidence_region()
        
        executionTime = (time.time() - t_pred)
        
        
        # Store forecast
        logger.info('Storing prediction data.')
        to_store = pd.DataFrame({'y':y_true,'y_hat': y_hat}, index = selected_instants)
        to_store.index.name = 'time'
        file_name = os.path.join(pred_path,'pred_{}_C{}_T{}.csv'.format(var,cycle_to_predict,n_instants))
        
        to_store.to_csv(file_name)
        
        
        plt_name = os.path.join(pred_path, 'pred_{}_C{}_T{}.png'.format(var,cycle_to_predict,n_instants))
    
        logger.info('Storing prediction plot.')
        plot_title = 'Cycle {} forecast VS observed'.format(cycle_to_predict)
        plot = plot_prediction(selected_instants, y_true, y_hat, title = plot_title, 
                               y_label = 'Observed', y_hat_label = 'Forecast',
                               x_axis_label = 't', y_axis_label = var, plt_name = plt_name)
        
        
        logger.info('Storing execution time.')
        times_df.loc[cycle_to_predict, 'time'] = executionTime
        
        
        time_file = os.path.join(output_path, 'time_T{}.csv'.format(n_instants))
        times_df.to_csv(time_file)
    
    logger.info('End.')
    logging.shutdown()


