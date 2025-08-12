# Functions which preprocess MRI images
# (c) 2025 Kamil Stachurski


import numpy as np

def apply_mask(img, mask):
    '''
    Apply a binary mask to an image (zeroing out values outside the mask).
    
    Args:
        img (np.ndarray): 3D or 4D MRI image
        mask (np.ndarray): 3D binary mask
        
    Returns:
        np.ndarray: Masked image.
    '''
    
    if img.ndim == 3:
        masked = img * mask
    elif img.ndim == 4:
        masked = img * mask[..., np.newaxis]
    else:
        raise ValueError('Image must be 3D or 4D')
    
    return masked