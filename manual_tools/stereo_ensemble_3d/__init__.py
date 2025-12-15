"""
Stereo Ensemble 3D PIV System for Reynolds Stress Extraction

This package provides tools for:
1. Generating synthetic stereo image pairs with known Reynolds stress tensors
2. MLOS (Multiplicative Line-of-Sight) volume reconstruction
3. 3D FFT-based ensemble correlation accumulation
4. 22-parameter stacked Gaussian fitting for displacement and Reynolds stress extraction

Usage:
    python -m stereo_ensemble_3d.run_stereo_ensemble --num-pairs 1000 --particles 500
"""

from .stereo_ensemble_generator import StereoEnsembleConfig, StereoEnsembleGenerator
from .correlation_3d import StereoMLOSReconstructor, EnsembleAccumulator3D
from .gaussian_fit_stacked_3d import fit_stacked_gaussian_3d, StackedGaussianResult3D

__all__ = [
    'StereoEnsembleConfig',
    'StereoEnsembleGenerator',
    'StereoMLOSReconstructor',
    'EnsembleAccumulator3D',
    'fit_stacked_gaussian_3d',
    'StackedGaussianResult3D',
]
