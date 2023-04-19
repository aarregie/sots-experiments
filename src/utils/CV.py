# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:51:27 2022

@author: aarregui
"""
import numpy as np
from src.utils.auxiliar import get_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

def GridSearchCV_TimeSeries(Y_matrix, an_type, param_grid, n_splits = 5, t_size = 1):
    '''
    Perform the grid search using CV for time series data.

    Parameters
    ----------
    Y_matrix : np.array (n_features, n_samples)
        Matrix containing the time series data.
    an_type : int
        Type of model to be trained.
    param_grid : ParameterGrid
        Parameter grid for the possible parameters.
    n_splits : int, optional
        Number of splits of the time series.. The default is 5.
    t_size : int, optional
        Test size for evaluating the forecasting performance of the models. The default is 1.

    Returns
    -------
    cv_result : dict
        Dictionary containing the best parameter combination, best scoring value
        and the best adjusted matrix.

    '''
    
    comb_param = list(param_grid)
    
    #Import specified model function
    model = get_model(an_type)
    logger.info('Model {} CV.'.format(an_type))
    
    #Initialize values
    actual_model = None
    best_mse = 1000
    
    #Grid search sobre los parámetros de los modelos
    logger.info('Loop over parameter combinations.')
    for comb in comb_param:
        
        logger.info('Comb: {}'.format(comb))
        spec_param, p = comb['param_1'], comb['p']
        
        #Split time series data
        tscv = TimeSeriesSplit(n_splits = n_splits, test_size = t_size)
        
        
        mse_cv = np.array([])
        
        for train, test in tscv.split(Y_matrix.T):
            logger.info('CV test {}'.format(str(test)))
            y_train = Y_matrix[:,train]
            y_test = Y_matrix[:, test].ravel()
            logger.debug('Fit model with selected parameter combination.')
            y_hat = model(y_train, p, spec_param)[0][:,-1].ravel()
            mse_fold = mean_squared_error(y_test, y_hat)
            mse_cv = np.append(mse_cv, mse_fold)
        
        comb_A = model(Y_matrix, p, spec_param)[1]
        mse_gs = mse_cv.mean()
            
        
        #Compare with previous mse and update best model
        if not actual_model:
            logger.debug('First parameter combination: {}'.format(comb))
            actual_model = comb
            best_mse = mse_gs
            A = comb_A
                
        else:
            if mse_gs <= best_mse:
                logger.debug('New best parameter combination: {}'.format(comb))
                best_mse = mse_gs
                actual_model = comb
                A = comb_A
    
    #Generate resulting dict
    cv_result = {'best_model': actual_model, 'best_score': mse_gs, 'best_params': A}
    
    
    return cv_result