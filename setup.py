from setuptools import setup, find_packages


setup(
    name = 'gmm-stroke',
    version = '0.1.0',
    description = 'GMM-based ischemic stroke segmentation from diffusion MRI',
    author = 'Kamil Stachurski',
    packages = find_packages(where = '.'),
    python_requires = '>=3.9',
    install_requires=[
        'numpy==2.3.2',
        'scipy==1.16.1',
        'scikit-learn==1.7.1',
        'matplotlib==3.10.5',
        'scikit-image==0.25.2',
        'pyvista==0.46.1',
        'nibabel==5.3.2',
        'brukerapi==0.1.9',
    ],
    entry_points={
        'console_scripts': [
            'gmm_stroke_predict=src.cli:main',
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
