# Advanced Gaussian Mixture Modeling for Ischemic Stroke MRI Analysis

(c) 2025 Kamil Stachurski


## 📖 Overview
GMM-Stroke is an open-source Python package designed for automatic detection and visualization of ischemic brain region based on diffusion coefficients in MRI data.
This project implements a novel image processing and analysis pipeline, which I developed during my PhD thesis, specifically for preclinical ischemic stroke models using high-field MRI. While primarily validated in animal models (preclinical), the methodology is also highly translatable to clinical neuroimaging, offering potential improvements in stroke diagnostics, treatment planning and outcome prediction.


## 💡 Key Innovation
* Introduces a custom Gaussian Mixture Model (GMM) fitting stratego tailored for ischemic stroke imaging
* Novel parameter initialization and processing steps designed to maximize sensitivity for ischemic core and penumbra detection
* Direct integration of preclinical Bruker MRI datasets with optional compatibility for NIfTI/DICOM formats
* Advanced 3D visualization of ischemic regions combined with structural MRI for anatomical context

This approach is original and authored by me (Kamil Stachurski), and it forms an integral part of my PhD dissertation.


## 🔬 Preclinical & Clinical Importance

### Preclinical Impact
* Provides researchers with a robust, reproducible, and automated method to segment ischemic lesions in rodent stroke models
* Facilitates quantitative analysis of stroke evolution, lesion volume, and treatment efficacy
* Compatible with Bruker ParaVision data, the gold standard in small animal MRI research

### Clinical Potential
* The methodology can be adapted for patient data (DICOM/NIfTI) in clinical stroke imaging
* Improves lesion characterization by separating ischemic core and penumbra based on ADC 
* Could enhance treatment decision-making in acute ischemic stroke by providing fast and reliable lesion maps


## 🧠 Pipeline Overview
1. Data Import - Bruker or NIfTI diffusion datasets
2. Brain Extraction - Apply precomputed brain masks
3. ADC Map Calculation - Convert DWI to ADC values, removing negative artifacts
4. GMM Fitting - Classify voxels into ischemic lesion, healthy tissue or cerebrospinal fluid
5. 3D Visualization - Render lesion masks over anatomical images


## 📦 Installation
```bash
git clone https://github.com/KamRoki/gmm_stroke.git
cd gmm_stroke
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```


## 📜 License
This project is licensed under a custom restrictive license – distribution, modification, and commercial use require explicit permission from the author.




