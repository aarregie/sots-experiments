

import numpy as np
from scipy.linalg import block_diag

def companion_form(X,p):
    
    '''
    Based on a multivariate time series matrix X, generates the explicative 
    and explanatory variable matrix of the companion form of the VAR(p) model,
    as in Lütkepohl, H. (2005), equation 3.2.2.
    Z = BY+U
    
    Parameters
    ----------
    X : np.array
        Matrix of size KxT, where K is the number of dimensions and T is the 
        length of the series.
    p : int
    Order of the VAR model.

    Returns
    -------
    Z : np.array
         Explicative matrix.
    Y : np.array
         Objective matrix.

    '''
    
    
    N, T = X.shape
    
    #Generate Z & Y
    Y = X[:, p :]
    
    Z = np.zeros((N*p, T-p))
    
    for t in range(T-p):
        vec = X[:, t:(t+p)].ravel()
        Z[:,t] = vec

    
    return Z, Y
    


def block_diagonal_form(X, p):
    '''
    This function composes the Y and Z and stackedblock diagonal matrix 
    of a VAR(p), that satisfies the formula
    Y = [A1,...,Ap]Z[I(N-p),...,I(N-p)]' 
    Further details, Hsu(2021)'

    Parameters
    ----------
    X : np.array
        Matrix of size (NxT) containing the whole time series data.
    p : int
        Order of the VAR model.

    Returns
    -------
    Z : np.array
        Explanatory variables matrix.
        It satisfies Z := Bdiag(Y_{t-1}, ..., Y_{t-p})
    Y : np.array
        Objective matrix. Y = {Y_{t-p+1}, ..., Y_{T}}
    stacked_I : np.array
        p times stacked diagonal matrix.

    '''
    
    T, N = X.shape
    #Compose Y
    Y = X[:, p :]    
    
    #Compose X
    Z = np.array([])
    
    for i in range(p):
        
        Y_i = X[:,(p-i-1):-(i+1)]
        if Z.any():
            Z = block_diag(Z,Y_i)
        else:
            Z = Y_i
        print(Z)

    
    return Z, Y


def bandwidth_mask(shape, L):
    '''
    Returns mask of non-zero elements of a SQUARED matrix with 
    the given shape.
    

    Parameters
    ----------
    shape : int
        Shape of the matrix--> (shape,shape)
    L : int
        Band width.

    Returns
    -------
    mask : np.array
        Mask of the banded matrix.

    '''

    mask = np.full((shape,shape), False)
    jj = np.arange(shape)
    
    for i in range(shape):
        col_mask = abs(jj - i) <= L
        mask[i,col_mask] = True
    
    return mask

    
