# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:46:56 2022

@author: aarregui
"""

import os


def check_existence(path):
    '''
    
    Check if given path exists.

    Parameters
    ----------
    path : string
        String containing the 

    Returns
    -------
    exists : Bool
        True/false if file exists/does not exist.
        
    '''
    
    exists = False
    if os.path.isdir(path):
        exists = True
    
    return exists

def make_dirs(path):
    '''
    
    Make the given dirs.

    Parameters
    ----------
    data : dict
        Cell data.

    Returns
    -------
    cycles_len : float
        Highest value of the instant.
    '''
    
    os.makedirs(path)
    return True


def validate_directory(path):
    '''
    
    Validate given directory. If the path does not exist, it's created.'

    Parameters
    ----------
    data : dict
        Cell data.

    Returns
    -------
    cycles_len : float
        Highest value of the instant.
    '''

    exists = check_existence(path)
    
    if not exists:
        make_dirs(path)
    
    return True

