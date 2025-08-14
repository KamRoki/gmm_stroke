# Functions which calculate ADC diffusion 
# (c) 2025 Kamil Stachurski


import numpy as np


def compute_adc(diff_img: np.ndarray,
                bvals: np.ndarray,
                brain_mask: np.ndarray) -> np.ndarray:
    '''
    Compute the Apparent Diffusion Coefficient (ADC) map from diffusion images.
    
    Args:
        diff_img (np.ndarray): Diffusion-weighted images, shape (x, y, z, n_bvals)
        bvals (np.ndarray): Array of b-values (s/mm²)
        brain_mask (np.ndarray): Binary mask of brain area, shape (x, y, z)
        
    Returns:
        np.ndarray: ADC map, shape (x, y, z)
    '''
    
    if diff_img.ndim != 4:
        raise ValueError('diff_img must be 4D (x, y, z, n_bvals)')
    
    if brain_mask.shape != diff_img.shape[:3]:
        raise ValueError('brain_mask shape does not match diff_img spatial dimensions')
    
    # Convert to float
    diff_img = diff_img.astype(np.float32)
    bvals = np.array(bvals,
                     dtype = np.float32)
    
    # Find b0 and A0 image
    b0_idx = np.argmin(bvals)
    A0 = diff_img[..., b0_idx]
    
    A0[A0 <= 0] = np.finfo(np.float32).eps
    
    # Prepare ADC map
    ADC_map = np.zeros(diff_img.shape[:3],
                       dtype = np.float32)
    
    # Fit slope for ln(S) = -b * ADC
    for x in range(diff_img.shape[0]):
        for y in range(diff_img.shape[1]):
            for z in range(diff_img.shape[2]):
                if not brain_mask[x, y, z]:
                    continue
                S = diff_img[x, y, z, :]
                if np.any(S <= 0):
                    S[S <= 0] = np.finfo(np.float32).eps
                ln_S_ratio = np.log(S / A0[x, y, z])
                slope, *_ = np.polyfit(bvals, ln_S_ratio, 1)
                ADC_map[x, y, z] = -slope
    
    # Set diffusion coefficients lower than 0
    ADC_map[ADC_map < 0] = 0.0
                
    return ADC_map