"""
Stereo Ensemble 3D PIV System for Reynolds Stress Extraction

This package provides tools for:
1. Generating synthetic stereo image pairs with known Reynolds stress tensors
2. MLOS (Multiplicative Line-of-Sight) volume reconstruction
3. 3D FFT-based ensemble correlation accumulation
4. 22-parameter stacked Gaussian fitting for displacement and Reynolds stress extraction
5. Transfer function method for Reynolds stress extraction (NEW)

Usage:
    python -m stereo_ensemble_3d.run_stereo_ensemble --num-pairs 1000 --particles 500
"""

from .stereo_ensemble_generator import StereoEnsembleConfig, StereoEnsembleGenerator
from .correlation_3d import StereoMLOSReconstructor, EnsembleAccumulator3D
from .gaussian_fit_stacked_3d import fit_stacked_gaussian_3d, StackedGaussianResult3D
from .transfer_function_3d import (
    compute_transfer_function,
    TransferFunctionResult,
    SpectralDivisionConfig
)
from .visualize_transfer_function import (
    create_transfer_function_comparison_figure,
    create_gaussianity_validation_figure,
    create_spectral_analysis_figure,
    create_reynolds_stress_comparison_figure,
    save_transfer_function_figures
)
from .gaussian_convolution_fit import (
    fit_gaussian_convolution,
    fit_gaussian_convolution_fourier,
    GaussianConvolutionResult,
    convolve_with_gaussian
)

__all__ = [
    # Original exports
    'StereoEnsembleConfig',
    'StereoEnsembleGenerator',
    'StereoMLOSReconstructor',
    'EnsembleAccumulator3D',
    'fit_stacked_gaussian_3d',
    'StackedGaussianResult3D',
    # New transfer function exports
    'compute_transfer_function',
    'TransferFunctionResult',
    'SpectralDivisionConfig',
    'create_transfer_function_comparison_figure',
    'create_gaussianity_validation_figure',
    'create_spectral_analysis_figure',
    'create_reynolds_stress_comparison_figure',
    'save_transfer_function_figures',
    # Gaussian convolution fitting (best method!)
    'fit_gaussian_convolution',
    'fit_gaussian_convolution_fourier',
    'GaussianConvolutionResult',
    'convolve_with_gaussian',
]
