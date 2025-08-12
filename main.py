# Main Python script of GMM ischemic extraction
# (c) 2025 Kamil Stachurski

import numpy as np
import matplotlib.pyplot as plt

from src.io import load_bruker_diffusion, load_mat_mask
from src.adc import compute_adc
from src.preprocessing import apply_mask
from src.modeling import fit_gmm_stroke
from src.visualization import plot_gmm_histogram, plot_3d_ischemia

def main():
    # Import data
    data_path = '/Users/kamil/Documents/my_softwares/gmm_stroke/data/rat_model_1/6/pdata/1/2dseq/'
    mask_path = '/Users/kamil/Documents/my_softwares/gmm_stroke/data/brain_mask.mat'
    
    diff_img, bvals = load_bruker_diffusion(data_path)
    brain_mask = load_mat_mask(mask_path)
    
    diff_img_masked = apply_mask(diff_img, brain_mask)
    print(f'Image imported: Shape = {diff_img_masked.shape}, Dtype = {diff_img_masked.dtype}')
    print(f'Number of bvals: {bvals.shape}')
    
    # Calculate ADC
    ADC_map = compute_adc(diff_img_masked, bvals, brain_mask)
    print(f'ADC map calculated! Shape: {ADC_map.shape}, Min: {ADC_map.min()}, Max: {ADC_map.max()}')
    
    # Visualize
    slice_idx = 22
    b0_idx = np.argmin(bvals)
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    
    axes[0].imshow(diff_img_masked[:, :, slice_idx, b0_idx], cmap = 'gray')
    axes[0].set_title(f'Diffusion image (A0) - slice {slice_idx}')
    axes[0].axis('off')
    
    im1 = axes[1].imshow(ADC_map[:, :, slice_idx] * 1e3, cmap='jet', vmin=0, vmax=1.2)
    axes[1].set_title(f"ADC map - slice {slice_idx}")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='ADC (x10⁻³ mm²/s)')

    plt.tight_layout()
    plt.show()
    
    adc_brain = ADC_map * brain_mask
    # --- 4) Fit GMM ---
    gmm, info = fit_gmm_stroke(adc_brain)

    # --- 5) Print fitted parameters ---
    print("\n===== GMM FIT RESULTS =====")
    print(f"Samples used: {info['n_samples']}")
    print(f"Range (mm²/s): {info['range_used_mm2s']}")
    print(f"Means (×1e-3 mm²/s): {info['means_x1e3']}")
    print(f"Variances: {info['vars_x1e3']}")
    print(f"Weights: {info['weights']}")
    
    # --- 6) Plot GMM results ---
    plot_gmm_histogram(adc_map = ADC_map,
                       model = gmm)
    
    # --- 7) Visualize 3D ischemic tissue in brain
    plot_3d_ischemia(diff_img = diff_img_masked,
                     adc_map = ADC_map,
                     model = gmm)
    
    
    


if __name__ == '__main__':
    main()