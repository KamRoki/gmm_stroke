# Functions which reads Bruker data
# (c) 2025 Kamil Stachurski

import numpy as np
import scipy.io as sio


try:
    from brukerapi.dataset import Dataset
except ImportError:
    print('Missing brukerapi import. Importing Bruker data will not be available')
    

def load_bruker_diffusion(data_path, method_file = 'method', reorder = True):
    '''
    Load Bruker diffusion data.
    
    Args:
        data_path (str): Path to the Bruker 2dseq data directory 
        method_file (str): Name of the method file to read parameters 
        reorder (bool): If True, reorder the image dimensions
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The diffusion-weighted image and the b-values.
    '''
    
    dataset = Dataset(data_path)
    img = dataset.data.squeeze()
    
    if reorder:
        img = np.rot90(np.transpose(img, (1, 0, 2, 3)), 2)
        
    dataset.add_parameter_file(method_file)
    bvals = np.array(dataset['PVM_DwEffBval'].value, dtype = float)
    
    return img, bvals


def load_mat_mask(mat_path, var_name = 'brain_mask'):
    '''
    Load a binary mask from a .mat file.

    Args:
        mat_path (str): Path to the .mat mask file (created in matlab)
        var_name (str): Name of the variable to load from the .mat file

    Returns:
        np.ndarray: The loaded binary mask.
    '''
    
    mat = sio.loadmat(mat_path)
    
    if var_name not in mat:
        raise KeyError(f"No variable '{var_name}' in file {mat_path}")
    
    return mat[var_name].astype(bool)