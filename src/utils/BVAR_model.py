# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:55:33 2022

@author: aarregui
"""

import numpy as np

from src.utils.matrix_operators import *

    
def BVAR(data, p, L, pred_step = 1):
    '''
    Banded VAR model learning.
    Model:
    Y = A@Z, where A is banded

    Parameters
    ----------
    data : np.array
        Multivariate time series of shape KxT, where K is the number of 
        dimensions and T is the length of the series.
    R : int
        rank of the reduced-rank form matrix.
    p : int
        order of the VAR model.
    pred_step : int
        Prediction step size. Default is 1.
    maxiter : int, optional
        Number of iterations to solve the problem. The default is 100.

    Returns
    -------
    pred : np.array
        Prediction matrix for the adjusted model.
    B : np.array
        Adjusted parameter matrix.

    '''
    

    N, T = data.shape
    
    #Generate Z & Y
    Z, Y = companion_form(data,p)


    A = estimate_params(Z, Y, p, L)
    
    pred = forecast(Y, A, Z, pred_step, p)
    return pred, A


def estimate_params(Z, Y, p, L):
    '''
    
    Estimator for banded VAR model parameters.

    Parameters
    ----------
    Z : np.array
        Explanatory time series.
    Y : np.array
        Objective time series.
    p : int
        Order of the VAR model.
    L : int
        Bandwidth of the parameter matrix.

    Returns
    -------
    A : np.array
        
    '''
    
    T = Y.shape[0]
    
    Q = bandwidth_mask(T,L)

    A = np.zeros((T,T*p))
    
    
    for t in range(T):
        
        
        row = Q[t,:]
        mask = np.array(list(row)*p)
            
        Y_t = Y[t,:].T
        X_t = Z[mask,:].T
        
        #LSE
        beta_i = np.linalg.pinv(X_t.T@X_t)@X_t.T@Y_t
        
        #Fill A matrix
        A[t,mask] = beta_i
        
    return A



def forecast(Y, A, Z, pred_step, p):
    '''
    
    Forecast time series observations.

    Parameters
    ----------
    Y : np.array
        Initial time series
    A : np.array
        Estimated VAR(p) model parameter matrix.
    Z : np.array
        Compact explanatory time series.
    pred_step : int
        Prediction step size.
    p : int
        VAR model order.

    Returns
    -------
    pred : np.array
        Matrix containing the predictions.

    '''
    
    N, T = Y.shape
    
    results_mat = np.append(Y,np.zeros((N, pred_step)), axis = 1)
    
    Z_new = np.append(Z, np.zeros((N*p, pred_step)), axis = 1)
    for s in range(pred_step):
        results_mat[:, T + s] = A @ Z_new[:, T + s - 1]

        #Update Z
        first_vec = Z_new[N:p*N, T+s-1].ravel()
        hat_vec = results_mat[:, T+s].ravel()
        Z_new[:, T+s] = np.append(first_vec, hat_vec)
    
    pred = results_mat[:, - pred_step :]
    
    return pred

