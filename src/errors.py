# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:36:22 2022

@author: aarregui
"""


import os
import pickle
import pandas as pd
import numpy as np
from src.utils import directories
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)


analysis = ['ARIMA', 'GP', 'BVAR', 'LRVAR', 'MTGP']

univariate = ['ARIMA','GP']
multivariate = ['BVAR', 'LRVAR' 'MTGP']


output_folder = os.path.join('output')
predictions_folder = 'predictions'
file_type = '.csv'


data_files = dict()

errors_dict = dict()


def errors(n_instants, var, cell_name):


    results_folder = os.path.join(output_folder, cell_name,'results')
    directories.validate_directory(results_folder)

    for an_type in analysis:
    
        if an_type in univariate:
            method = 'univariate'
        elif an_type in multivariate:
            method = 'multivariate'
    
        
        df_et = pd.DataFrame({})
        #Generate output path string
        output_path = os.path.join(output_folder, cell_name, method, an_type, var)
        pred_path = os.path.join(output_path, predictions_folder)
        
        if os.path.isdir(pred_path):
            files = os.listdir(pred_path)
            if len(files)>0:
                
                #filter by n_instants
                files = [file for file in files if 'T{}'.format(n_instants) in file]
                
                data_files_an = [file for file in files if file_type in file]
                
                predicted_cycles = [file.split('_')[2].replace(file_type,'') for file in data_files_an]
                
                predicted_cycles = [int(cycle.replace('C','')) for cycle in predicted_cycles]
                
                for file_name in data_files_an:
    
                    #t = cell[c][str(pc)]['t']
                    pred_df = pd.read_csv(os.path.join(pred_path, file_name), index_col = 0)
                    
                    y = pred_df['y'].values
                    y_hat = pred_df['y_hat'].values
                    
                    #mean of the true series y
                    y_maxmin = y.max()-y.min()
                    
                   
                    mse = mean_squared_error(y, y_hat)
                    rmse = np.sqrt(mse)
                    nrmse = rmse/y_maxmin
         
                    pred_results = {'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse}
                    df_et = df_et.append(pred_results, ignore_index = True)
                
                df_et['cycles'] = predicted_cycles
                df_et = df_et.set_index('cycles')
                df_et = df_et.sort_index()
                errors_dict[an_type] = df_et
                
                #for plotting 
    
    file_name = os.path.join('output', 'general_results', '{}_errors_{}_T{}.pickle'.format(cell_name, var, n_instants))
    
    with open(file_name, 'wb') as handle:
        pickle.dump(errors_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    

