# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:55:33 2022

@author: aarregui
"""

import numpy as np

from src.utils.matrix_operators import companion_form

    
def LRVAR(data, p, R, pred_step = 1, maxiter = 100):
    '''
    Reduced-rank VAR algorithm.
    Model:
    Y = W@V@Z

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
        Prediction step size.
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


    W, V = estimate_params(Z, Y, R, maxiter)
    B = W@V
    #mse = compute_var_error(Z, Y, B)
    
    pred = forecast(Y, B, Z, pred_step, p)
    return pred, B


def estimate_params(Z, Y, R, maxiter):
    '''
    
    Iterative least squares for estimating W, V, so that
    Y = W@V@Z

    Parameters
    ----------
    Z : np.array
        Explanatory time series.
    Y : np.array
        Objective time series.
    R : int
        Rank of the reduced form of the matrix B.
    maxiter : int
        Iteration number.
    Returns
    -------
    W : np.array
        
    V : np.array
        

    '''
    
    #Define V matrix shape
    n_cols = Z.shape[0]
    V = np.random.randn(R, n_cols)
    #Iterative algorithm
    for it in range(maxiter):
        W = Y @ np.linalg.pinv(V @ Z)
        V = np.linalg.pinv(W) @ Y @ np.linalg.pinv(Z)
    return W, V


def forecast(Y, B, Z, pred_step, p):
    '''
    
    Forecast time series observations.

    Parameters
    ----------
    Y : np.array
        Initial time series
    B : np.array
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
        Matrix containing the pred_step predictions.

    '''
    
    N, T = Y.shape
    
    results_mat = np.append(Y,np.zeros((N, pred_step)), axis = 1)

    Z_new = np.append(Z, np.zeros((N*p, pred_step)), axis = 1)
    for s in range(pred_step):
        results_mat[:, T + s] = B @ Z_new[:, T + s - 1]

        #Update Z
        first_vec = Z_new[N:p*N, T+s-1].ravel()
        hat_vec = results_mat[:, T+s].ravel()
        Z_new[:, T+s] = np.append(first_vec, hat_vec)

    pred = results_mat[:, - pred_step :]
    
    return pred
