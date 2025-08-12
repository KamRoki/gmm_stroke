# Functions which visualize the results
# (c) 2025 Kamil Stachurski


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from typing import Tuple
from skimage.morphology import remove_small_objects, binary_closing, ball
import pyvista as pv


def plot_gmm_histogram(
    adc_map: np.ndarray,
    model: GaussianMixture,
    lower_bound: float = 0.2,
    upper_bound: float = 1.4,
):
    
    lower_diff_threshold = lower_bound * 1e-3
    upper_diff_threshold = upper_bound * 1e-3
    
    adc_vals = adc_map[
        (adc_map >= lower_diff_threshold) & (adc_map <= upper_diff_threshold)
    ]
    
    counts, bins, patches = plt.hist(adc_vals * 1e3,
                                     bins = np.arange(lower_bound, upper_bound, 0.01),
                                     density = False,
                                     alpha = 0.6,
                                     color = 'lightblue',
                                     edgecolor = 'black',
                                     label = 'ADC Histogram')
    
    x_range = np.linspace(lower_bound, upper_bound, 1000)
    x_range_reshaped = x_range.reshape(-1, 1)
    
    colors = ['red', 'green', 'blue']
    component_names = ['Ischemic Tissue', 'Healthy Tissue', 'CBF']
    
    total_samples = len(adc_vals)
    bin_width = bins[1] - bins[0]
    
    for i in range(model.n_components):
        component_pdf = model.weights_[i] * norm.pdf(
            x_range,
            model.means_[i, 0],
            np.sqrt(model.covariances_[i, 0])
        )
        component_counts = component_pdf * total_samples * bin_width
        plt.plot(
            x_range,
            component_counts,
            color = colors[i],
            linewidth = 2,
            linestyle = '--',
            label = f'{component_names[i]} (μ={model.means_[i, 0]:.2f}, w={model.weights_[i]:.2f})'
        )
        
    total_pdf = np.exp(model.score_samples(x_range_reshaped))
    total_counts = total_pdf * total_samples * bin_width
    plt.plot(x_range,
             total_counts,
             color = 'black',
             linewidth = 3,
             label = 'Total GMM Fit')
    
    plt.xlabel('ADC Values (mm²/s × 1e-3)')
    plt.ylabel('Count')
    plt.title('ADC Histogram with Fitted Gaussian Mixture Model')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.xlim(lower_bound, upper_bound)
    plt.tight_layout()
    plt.show()
    
    
def plot_3d_ischemia(
    diff_img: np.ndarray,
    adc_map: np.ndarray,
    model: GaussianMixture,
    confidence: float = 0.7,
    spacing: Tuple[float, float, float] = (0.267, 0.333, 0.5)
):
    X_all_mm = (adc_map * 1e3).reshape(-1, 1)
    probs = model.predict_proba(X_all_mm)
    ischemic_class = np.argmin(model.means_.ravel())
    
    max_class = np.argmax(probs, axis = 1)
    mask_flat = (max_class == ischemic_class) & (probs[:, ischemic_class] > confidence)
    ischemia_mask = mask_flat.reshape(adc_map.shape)
    
    ischemia_mask = remove_small_objects(ischemia_mask, min_size = 100) # voxele <100 out
    ischemia_mask = binary_closing(ischemia_mask, footprint = ball(1))  # close small holes
    
    b0 = diff_img[..., 0].astype(np.float32)
    b0 /= (b0.max() + 1e-12)
    mask_vis = ischemia_mask.astype(np.uint8)
    
    origin = (0.0, 0.0, 0.0)
    nx, ny, nz = b0.shape
    
    grid = pv.StructuredGrid()
    x = np.arange(0, nx * spacing[0], spacing[0])
    y = np.arange(0, ny * spacing[1], spacing[1])
    z = np.arange(0, nz * spacing[2], spacing[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
    grid.points = np.c_[xx.ravel(order = 'F'), yy.ravel(order = 'F'), zz.ravel(order = 'F')]
    grid.dimensions = (nx, ny, nz)
    grid['intensity'] = b0.ravel(order = 'F')
    
    mask_grid = pv.StructuredGrid()
    mask_grid.points = np.c_[xx.ravel(order = 'F'), yy.ravel(order = 'F'), zz.ravel(order = 'F')]
    mask_grid.dimensions = (nx, ny, nz)
    mask_grid['values'] = mask_vis.ravel(order = 'F')
    
    contours = mask_grid.contour(isosurfaces = [0.5])
    
    plotter = pv.Plotter(notebook = False, window_size = (1200, 800))
    plotter.add_volume(grid,
                       cmap = 'gray',
                       opacity = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 1.00],
                       shade = False,
                       show_scalar_bar = False)
    plotter.add_mesh(contours, color = 'red', opacity = 0.6, smooth_shading = True)
    plotter.add_axes()
    plotter.show()