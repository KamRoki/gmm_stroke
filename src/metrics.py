from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture


def _voxel_mm3(spacing_mm: tuple[float, float, float]) -> float:
    dx, dy, dz = spacing_mm
    return float(dx) * float(dy) * float(dz)


def calculate_stroke_volume(
    adc_map: np.ndarray,
    gmm: sklearn.mixture.GaussianMixture,
    spacing_mm: tuple[float, float, float],
    brain_mask: np.ndarray | None = None
) -> float:
    '''
    Return the expected ischemic lesion volume [mm3] from GMM posteriors.
    '''
    if brain_mask is not None:
        adc_use = adc_map * brain_mask
    else:
        adc_use = adc_map
        
    means = gmm.means_.ravel()
    ischemic_class = int(np.argmin(means))
    
    X_all_mm = (adc_use * 1e3).reshape(-1, 1)
    p_stroke = gmm.predict_proba(X_all_mm)[:, ischemic_class].reshape(adc_use.shape)
    
    if brain_mask is not None:
        p_stroke = p_stroke * brain_mask
        
    v_mm3 = _voxel_mm3(spacing_mm) * np.sum(p_stroke, dtype = np.float64)
    
    print(f'Stroke Volume is {v_mm3} mm3.')
    