# Functions which fit model of ADC diffusion
# (c) 2025 Kamil Stachurski


import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple, Dict


def fit_gmm_stroke(
    adc_map: np.ndarray,
    n_components: int = 3,
    means_init: Tuple[float, ...] = (0.35, 0.60, 0.90),
    weights_init: Tuple[float, ...] = (0.2, 0.7, 0.1),
    covariance_type: str = 'diag',
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    random_state: int = 42,
    value_range: Tuple[float, float] = (2e-4, 1.4e-3)
) -> Tuple[GaussianMixture, Dict]:
    '''
    Fit a Gaussian Mixture Model (GMM) to ADC values (in mm²/s) from an already extracted brain region.    
    
    This function:
    1. Flattens the ADC map to 1D values.
    2. Filters values within a specified range (default: 0.0002–0.0014 mm²/s).
    3. Converts units to mm²/s × 1e-3 for numerical stability in EM fitting.
    4. Initializes GMM parameters (means, weights).
    5. Fits the GMM and returns both the trained model and diagnostic information.
    
    Args:
        adc_map : np.ndarray
            ADC map (already masked to brain region) in mm²/s.
        n_components : int
            Number of Gaussian components to fit.
        means_init_mm : tuple of floats
            Initial means for GMM components, in mm²/s × 1e-3.
        weights_init : tuple of floats
            Initial weights for GMM components.
        covariance_type : str
            Covariance type for GMM ('full', 'tied', 'diag', 'spherical').
        reg_covar : float
            Non-negative regularization added to the diagonal of covariance matrices.
        max_iter : int
            Maximum number of EM iterations.
        random_state : int
            Random seed for reproducibility.
        value_range : tuple of floats
            Lower and upper bounds for ADC values (in mm²/s) to include in fitting.
            
    Returns:
        gmm : GaussianMixture
            The fitted GaussianMixture model.
        info : dict
            Dictionary containing:
                - n_samples: number of samples used in fitting
                - range_used_mm2s: (lower, upper) ADC range in mm²/s
                - means_x1e3: fitted means in mm²/s × 1e-3
                - vars_x1e3: fitted variances in mm²/s × 1e-3
                - weights: fitted mixture weights
    '''
    
    
    # Prepare data for training
    X = adc_map.flatten()
    lower_bound, upper_bound = value_range
    X = X[
        (X >= lower_bound) & (X <= upper_bound)
    ]
    if X.size == 0:
        raise ValueError('None ADC values in this range')
    
    # Scale and reshaping
    X_mm = (X * 1e3).reshape(-1, 1)
    
    # Modeling GMM
    gmm = GaussianMixture(
        n_components = n_components,
        covariance_type = covariance_type,
        reg_covar = reg_covar,
        max_iter = max_iter,
        random_state = random_state,
        means_init = np.array(means_init).reshape(-1, 1),
        weights_init = np.array(weights_init)
    )
    
    gmm.fit(X_mm)
    
    # Get info
    info = {
        'n_samples': X_mm.shape[0],
        'range_used_mm2s': value_range,
        'means_x1e3': gmm.means_.ravel(),
        'vars_x1e3': gmm.covariances_.ravel(),
        'weights': gmm.weights_
    }
    
    return gmm, info
    