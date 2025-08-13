from setuptools import setup, find_packages

setup(
    name="gmm-stroke",
    version="0.1.0",
    description="GMM-based ischemic stroke segmentation from diffusion MRI",
    author="Kamil Stachurski",
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "scikit-image",
        "pyvista",
        "nibabel",
        "brukerapi",
    ],
    entry_points={
        "console_scripts": [
            "gmm_stroke_predict=src.cli:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
