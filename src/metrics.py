from __future__ import annotations
import numpy as np


def _voxel_mm3(spacing_mm: tuple[float, float, float]) -> float:
    dx, dy, dz = spacing_mm
    return float(dx) * float(dy) * float(dz)


def probabilistic_stroke_metrics(
    adc_map: np.ndarray,
    gmm,
    spacing_mm: tuple[float, float, float],
    brain_mask: np.ndarray) -> dict:
    pass

