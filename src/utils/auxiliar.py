"""
Created on Wed Apr  6 09:51:27 2022

@author: aarregui
"""
import importlib


def get_interpolator(an_type):
    '''
    Gets the interpolation function that contains the an_type.

    Parameters
    ----------
    an_type : string
        Analysis type.

    Returns
    -------
    interpolator : dict
        Dictionary containing all the interpolation functions.

    '''
    
    interpolator = an_type.split('_')[0]
    return interpolator



def dict_filter(dict_to_filter, keys):
    '''
    Selects the items of the given key values.

    Parameters
    ----------
    dict_to_filter : dict
        Dictionary to filter.
    keys : list
        List of key values to be kept.

    Returns
    -------
    filtered_dict : dict
        Filtered dict.

    '''
    
    filtered_dict = dict([ (i,dict_to_filter[i]) for i in keys if i in set(dict_to_filter)])
    return filtered_dict



def get_model(model_type, module_path = 'src.utils'):
    '''
    Imports the module defined in module_path, given a key string.

    Parameters
    ----------
    model_type : str
        String that contains the name of the module to import.
    module_path : string, optional
        DESCRIPTION. The default is 'utils'.

    Returns
    -------
    bool
        Whether the module has been imported or not.

    '''
    
    name = '{}.{}_model'.format(module_path, model_type)
    pos_import = importlib.util.find_spec(name)
    
    if pos_import is None:
        return False
    
    mod = importlib.import_module(name)
    model = getattr(mod, model_type)
    
    return model