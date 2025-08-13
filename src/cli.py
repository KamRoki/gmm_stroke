import argparse
import numpy as np
from src.io import load_bruker_diffusion, load_mat_mask
from src.preprocessing import apply_mask
from src.adc import compute_adc
from src.modeling import fit_gmm_stroke
from src.visualization import plot_gmm_histogram, plot_3d_ischemia


def main():
    parser = argparse.ArgumentParser(
        description = 'GMM-Stroke: Automatic ischemic lesion detection from DWI/ADC magnetic resonance imaging'
    )
    
    parser.add_argument(
        '-data', '--data_path',
        required = True,
        help = 'Path to Bruker \'2dseq\' data'
    )
    
    parser.add_argument(
        '-mask', '--mask_path',
        required = True,
        help = 'Path to brain mask'
    )
    
    args = parser.parse_args()
    
    print('[INFO] Loading data...')
    diff_img, bvals = load_bruker_diffusion(args.data_path)
    brain_mask = load_mat_mask(args.mask_path)
    diff_img_masked = apply_mask(diff_img, brain_mask)
    
    print('[INFO] Computing ADC map...')
    ADC_map = compute_adc(diff_img, bvals, brain_mask)
    ADC_brain = ADC_map * brain_mask
    
    print('[INFO] Fitting GMM model...')
    gmm, info = fit_gmm_stroke(ADC_brain)
    
    print('\n===== GMM PARAMETERS =====')
    means = info["means_x1e3"]     # w mm²/s × 1e-3
    vars_ = info["vars_x1e3"]      # wariancje w mm²/s × 1e-3
    weights = info["weights"]
    for i, (mu, var, w) in enumerate(zip(means, vars_, weights), start=1):
        print(f"Component {i}: μ={mu:.3f} (×1e-3 mm²/s), σ={np.sqrt(var):.3f}, w={w:.3f}")
        
    print('[INFO] Plotting histogram...')
    plot_gmm_histogram(ADC_brain, gmm)
    
    print('[INFO] Visulizing 3D of ischemia')
    plot_3d_ischemia(diff_img_masked, ADC_map, gmm)
    

if __name__ == '__main__':
    main()