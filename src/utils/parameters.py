
from sklearn.model_selection import ParameterGrid
import logging

logger = logging.getLogger(__name__)

def init_param_grid(an_type = None):
    '''
    Initialize param grid depending on the analysis type to be performed.

    Parameters
    ----------
    an_type : int
        Type of model to be adjusted.

    Returns
    -------
    param_grid : dict
        Parameter grid.

    '''
    logger.info('Initializing parameter grid.')
    logger.info('Model type: {}'.format(an_type))
    if an_type == 'ARIMA':
        param_grid = {'param_1': [0, 1, 2, 3, 4], 'p': [1, 2, 3, 4, 5]} 
    elif an_type=='LRVAR':
        param_grid = {'param_1': [2, 3, 4, 5, 6], 'p': [1, 2, 3, 4, 5]}
        
        
    elif an_type == 'BVAR':
        param_grid = { 'param_1': [1, 2, 3, 4, 5], 'p': [1, 2, 3, 4, 5]}
                        
    else:
        logger.info('Generic parameter grid.')
        #initialize generic param grid with all the possible parameters
        param_grid = {'param_1': [2, 3, 4, 5, 6], 'p': [1, 2, 3, 4, 5]}
    
    
    return param_grid



def check_param_grid(param_grid, T, an_type = None):
    '''
    Checks if the values of the grid for the parameter p (order of the VAR model)
    are suitable for the number of observations.

    Parameters
    ----------
    param_grid : dict
        Param grid.
    T : int
        Length of the time series.
    an_type : str, optional
        Model type. The default is None.

    Returns
    -------
    param_grid : dict
        Param grid suitable for the number of observations.

    '''
    logger.info('Checking if grid is valid...')
    if param_grid:
        
        p_array = param_grid['p']
        valid_ps = []            
        for p in p_array:
            if p<T:
                valid_ps.append(p)
        if len(valid_ps) != len(p_array):
            logger.info('Values for p are reduced to be able to fit the model.')
        
        param_grid['p'] = valid_ps
        
    
    return param_grid


def generate_param_grid(an_type, T):
    '''
    Automatic generation of a valid parameter grid based on the given analysis type.

    Parameters
    ----------
    an_type : string
        Type of analysis to be performed. The parameters are generated based
        on this value.
    T : int
        Time series length.

    Returns
    -------
    comb_param : ParameterGrid
        The different combinations of the parameters.

    '''
    
    param_grid = init_param_grid(an_type)
    
    param_grid = check_param_grid(param_grid, T, an_type)
    param_grid = ParameterGrid(param_grid)
    
    return param_grid
    
    