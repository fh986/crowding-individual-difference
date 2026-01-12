"""
Correlation Attenuation Toolbox
================================

A Python toolbox for correcting correlation coefficients for attenuation
due to measurement unreliability.

This package provides tools for:
1. Analyzing existing data with test-retest measurements
2. Planning future studies through simulation

Key Features
------------
- Computes naive and attenuation-corrected correlations
- Bootstrap confidence intervals for all estimates
- Monte Carlo simulations for power analysis
- Comprehensive visualization functions
- Support for multiple tasks with repeated measurements

Quick Start
-----------
For existing data analysis:

    >>> from attenuation_toolbox import analyze_correlation
    >>> import pandas as pd
    >>> 
    >>> # Your data: subjects x repeated measurements for each task
    >>> task1_data = pd.DataFrame(...)  # shape: (n_subjects, n_repeats)
    >>> task2_data = pd.DataFrame(...)
    >>> 
    >>> result = analyze_correlation(task1_data, task2_data)
    >>> print(result)

For study planning:

    >>> from attenuation_toolbox import TaskParameters, run_simulation_grid
    >>> 
    >>> task1 = TaskParameters(between_var=0.03, within_var=0.02, mean=0)
    >>> task2 = TaskParameters(between_var=0.05, within_var=0.01, mean=0)
    >>> 
    >>> results = run_simulation_grid(task1, task2, true_correlation=0.6)
    >>> print(results)

References
----------
- Spearman, C. (1904). The proof and measurement of association between
  two things. American Journal of Psychology, 15, 72-101.
- Diedrichsen, J., & Shadmehr, R. (2005). Detecting and adjusting for
  artifacts in fMRI time series data. NeuroImage, 27(3), 624-634.
"""

__version__ = '1.0.0'
__author__ = 'Correlation Attenuation Toolbox Contributors'

# Core analysis functions
from .core import (
    TaskParameters,
    CorrelationResult,
    compute_variances,
    compute_reliability,
    compute_reliability_split_half,
    correct_correlation,
    analyze_correlation,
    analyze_all_pairs,
    extract_task_measurements,
    summarize_task_statistics,
)

# Simulation functions
from .simulation import (
    SimulationResult,
    simulate_test_retest_data,
    simulate_multi_task_data,
    run_simulation,
    run_simulation_grid,
    get_simulation_distributions,
    estimate_required_sample_size,
    power_analysis_correlation,
)

# Visualization functions
from .visualization import (
    set_publication_style,
    plot_correlation_heatmaps,
    plot_correlation_histograms,
    plot_rmse_comparison,
    plot_bias_variance_comparison,
    plot_scatter_with_reliability,
    plot_simulation_summary,
    plot_reliability_effect,
)

# Convenience list of all public functions
__all__ = [
    # Core
    'TaskParameters',
    'CorrelationResult',
    'compute_variances',
    'compute_reliability',
    'compute_reliability_split_half',
    'correct_correlation',
    'analyze_correlation',
    'analyze_all_pairs',
    'extract_task_measurements',
    'summarize_task_statistics',
    # Simulation
    'SimulationResult',
    'simulate_test_retest_data',
    'simulate_multi_task_data',
    'run_simulation',
    'run_simulation_grid',
    'get_simulation_distributions',
    'estimate_required_sample_size',
    'power_analysis_correlation',
    # Visualization
    'set_publication_style',
    'plot_correlation_heatmaps',
    'plot_correlation_histograms',
    'plot_rmse_comparison',
    'plot_bias_variance_comparison',
    'plot_scatter_with_reliability',
    'plot_simulation_summary',
    'plot_reliability_effect',
]
