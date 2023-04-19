

import yaml

def parse_config(cfg_path):
    '''
    Load configuration file.

    Parameters
    ----------
    cfg_path : String
        String containing the path to the config yaml file.

    Returns
    -------
    config : dict
        Dictionary containing the configuration of the experimentation.

    '''
    
    with open(cfg_path, "r") as f:
        config = yaml.full_load(f)
    f.close()
    
    return config


def convert_value(config, key, init_value, final_value):
    '''
    Modify float values in configuration dictionary.

    Parameters
    ----------
    config : Dict
        String containing the path to the config yaml file.
    
    key : String
        String containing the key of the dictionary to be changed.
    
    init_value : object
        Value to be modified in config[key]. For checking purposes.
    
    final_value: object
        Final value in config[key].
    
    Returns
    -------
    config : dict
        Dictionary containing the configuration of the experimentation with the modified values in the given key.

    '''
    
    if config[key] == init_value:
        config[key] = final_value
    
    return config


    
    