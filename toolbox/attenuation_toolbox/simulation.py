"""
Correlation Attenuation Toolbox - Simulation Module

This module provides functions for:
1. Simulating data with known correlation and measurement noise
2. Running Monte Carlo simulations to compare estimators
3. Power analysis for study planning

These tools help researchers plan future studies by determining:
- Required sample sizes
- Number of repeated measurements needed
- Expected accuracy of correlation estimates
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

from .core import (
    TaskParameters, 
    compute_reliability, 
    correct_correlation,
    CorrelationResult
)


@dataclass
class SimulationResult:
    """
    Results from a Monte Carlo simulation.
    
    Attributes:
        n_subjects: Sample size used
        n_repeats: Number of repeated measurements per task
        true_correlation: Ground truth correlation
        n_iterations: Number of simulation iterations
        
        naive_mean: Mean of naive correlation estimates
        naive_sd: SD of naive correlation estimates
        naive_rmse: RMSE of naive estimates relative to true correlation
        naive_ci: 95% CI from simulation distribution
        
        corrected_mean: Mean of corrected correlation estimates
        corrected_sd: SD of corrected correlation estimates
        corrected_rmse: RMSE of corrected estimates
        corrected_ci: 95% CI from simulation distribution
        
        recommended_estimator: Which estimator has lower RMSE
    """
    n_subjects: int
    n_repeats: int
    true_correlation: float
    n_iterations: int
    
    naive_mean: float
    naive_sd: float
    naive_rmse: float
    naive_ci: Tuple[float, float]
    
    corrected_mean: float
    corrected_sd: float
    corrected_rmse: float
    corrected_ci: Tuple[float, float]
    
    recommended_estimator: str
    
    def __str__(self):
        return (
            f"Simulation Results (N={self.n_subjects}, repeats={self.n_repeats})\n"
            f"{'='*60}\n"
            f"True correlation: {self.true_correlation:.4f}\n"
            f"Iterations: {self.n_iterations}\n\n"
            f"Naive Estimator:\n"
            f"  Mean: {self.naive_mean:.4f}  SD: {self.naive_sd:.4f}\n"
            f"  RMSE: {self.naive_rmse:.4f}  95% CI: [{self.naive_ci[0]:.3f}, {self.naive_ci[1]:.3f}]\n\n"
            f"Corrected Estimator:\n"
            f"  Mean: {self.corrected_mean:.4f}  SD: {self.corrected_sd:.4f}\n"
            f"  RMSE: {self.corrected_rmse:.4f}  95% CI: [{self.corrected_ci[0]:.3f}, {self.corrected_ci[1]:.3f}]\n\n"
            f"Recommended estimator: {self.recommended_estimator}\n"
        )


def simulate_test_retest_data(
    task1_params: TaskParameters,
    task2_params: TaskParameters,
    n_subjects: int = 100,
    n_repeats: int = 4,
    true_correlation: float = 0.6,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate test-retest data for two correlated tasks with measurement noise.
    
    Generates data where each subject has a "true" latent score on each task,
    and multiple noisy measurements of each task are observed.
    
    Parameters
    ----------
    task1_params : TaskParameters
        Parameters for task 1 (between_var, within_var, mean).
    task2_params : TaskParameters
        Parameters for task 2.
    n_subjects : int, default=100
        Number of subjects to simulate.
    n_repeats : int, default=4
        Number of repeated measurements per task.
    true_correlation : float, default=0.6
        True (latent) correlation between tasks.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Simulated data with columns:
        - task1_true, task2_true: Latent true scores
        - task1_repeat1, task1_repeat2, ...: Noisy measurements for task 1
        - task2_repeat1, task2_repeat2, ...: Noisy measurements for task 2
        - task1_mean, task2_mean: Mean across repeats
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create covariance matrix for true scores
    cov_xy = true_correlation * np.sqrt(task1_params.between_var * task2_params.between_var)
    cov_matrix = np.array([
        [task1_params.between_var, cov_xy],
        [cov_xy, task2_params.between_var]
    ])
    
    # Simulate true latent scores
    true_scores = np.random.multivariate_normal(
        mean=[task1_params.mean, task2_params.mean],
        cov=cov_matrix,
        size=n_subjects
    )
    
    true_task1 = true_scores[:, 0]
    true_task2 = true_scores[:, 1]
    
    # Initialize data dictionary
    data = {
        'task1_true': true_task1,
        'task2_true': true_task2
    }
    
    # Generate noisy repeated measurements
    task1_measurements = []
    task2_measurements = []
    
    for i in range(n_repeats):
        # Add measurement noise
        noise1 = np.random.normal(0, np.sqrt(task1_params.within_var), n_subjects)
        noise2 = np.random.normal(0, np.sqrt(task2_params.within_var), n_subjects)
        
        measurement1 = true_task1 + noise1
        measurement2 = true_task2 + noise2
        
        data[f'task1_repeat{i+1}'] = measurement1
        data[f'task2_repeat{i+1}'] = measurement2
        
        task1_measurements.append(measurement1)
        task2_measurements.append(measurement2)
    
    # Add means across repeats
    data['task1_mean'] = np.mean(task1_measurements, axis=0)
    data['task2_mean'] = np.mean(task2_measurements, axis=0)
    
    return pd.DataFrame(data)


def simulate_multi_task_data(
    task_params_list: List[TaskParameters],
    correlation_matrix: np.ndarray,
    n_subjects: int = 100,
    n_repeats: int = 4,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate test-retest data for multiple correlated tasks.
    
    Parameters
    ----------
    task_params_list : list of TaskParameters
        Parameters for each task.
    correlation_matrix : np.ndarray
        Matrix of true correlations between tasks.
    n_subjects : int
        Number of subjects.
    n_repeats : int
        Number of repeated measurements per task.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Simulated data with columns for each task's repeats.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_tasks = len(task_params_list)
    
    # Build covariance matrix from correlations and between-subject variances
    between_vars = np.array([p.between_var for p in task_params_list])
    within_vars = np.array([p.within_var for p in task_params_list])
    means = np.array([p.mean for p in task_params_list])
    
    # Convert correlation matrix to covariance matrix
    sd_matrix = np.sqrt(np.outer(between_vars, between_vars))
    cov_matrix = correlation_matrix * sd_matrix
    
    # Simulate true scores
    true_scores = np.random.multivariate_normal(
        mean=means,
        cov=cov_matrix,
        size=n_subjects
    )
    
    # Build data dictionary
    data = {}
    for t, params in enumerate(task_params_list):
        task_name = params.name
        data[f'{task_name}_true'] = true_scores[:, t]
        
        measurements = []
        for r in range(n_repeats):
            noise = np.random.normal(0, np.sqrt(params.within_var), n_subjects)
            measurement = true_scores[:, t] + noise
            data[f'{task_name}_repeat{r+1}'] = measurement
            measurements.append(measurement)
        
        data[f'{task_name}_mean'] = np.mean(measurements, axis=0)
    
    return pd.DataFrame(data)


def run_simulation(
    task1_params: TaskParameters,
    task2_params: TaskParameters,
    n_subjects: int = 100,
    n_repeats: int = 4,
    true_correlation: float = 0.6,
    n_iterations: int = 1000,
    random_state: Optional[int] = None
) -> SimulationResult:
    """
    Run Monte Carlo simulation to evaluate correlation estimators.
    
    Repeatedly simulates data and computes both naive and corrected
    correlations to characterize their sampling distributions.
    
    Parameters
    ----------
    task1_params : TaskParameters
        Parameters for task 1.
    task2_params : TaskParameters
        Parameters for task 2.
    n_subjects : int
        Sample size per simulation.
    n_repeats : int
        Number of repeated measurements per task.
    true_correlation : float
        Ground truth correlation.
    n_iterations : int
        Number of Monte Carlo iterations.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    SimulationResult
        Summary of simulation results.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    naive_estimates = []
    corrected_estimates = []
    
    for _ in range(n_iterations):
        # Simulate dataset
        sim_data = simulate_test_retest_data(
            task1_params, task2_params,
            n_subjects=n_subjects,
            n_repeats=n_repeats,
            true_correlation=true_correlation
        )
        
        # Compute naive correlation (using means)
        naive_r, _ = pearsonr(sim_data['task1_mean'], sim_data['task2_mean'])
        
        # Compute reliabilities
        task1_cols = [f'task1_repeat{i+1}' for i in range(n_repeats)]
        task2_cols = [f'task2_repeat{i+1}' for i in range(n_repeats)]
        
        rel_task1 = compute_reliability(sim_data[task1_cols])
        rel_task2 = compute_reliability(sim_data[task2_cols])
        
        # Compute corrected correlation
        attenuation = np.sqrt(rel_task1 * rel_task2)
        corrected_r = naive_r / attenuation if attenuation > 0 else np.nan
        
        naive_estimates.append(naive_r)
        corrected_estimates.append(corrected_r)
    
    # Convert to arrays and filter NaNs for corrected
    naive_arr = np.array(naive_estimates)
    corrected_arr = np.array(corrected_estimates)
    corrected_valid = corrected_arr[~np.isnan(corrected_arr)]
    
    # Compute summary statistics
    def rmse(estimates, true_val):
        return np.sqrt(np.mean((estimates - true_val) ** 2))
    
    naive_mean = np.mean(naive_arr)
    naive_sd = np.std(naive_arr)
    naive_rmse = rmse(naive_arr, true_correlation)
    naive_ci = (np.percentile(naive_arr, 2.5), np.percentile(naive_arr, 97.5))
    
    corrected_mean = np.mean(corrected_valid)
    corrected_sd = np.std(corrected_valid)
    corrected_rmse = rmse(corrected_valid, true_correlation)
    corrected_ci = (np.percentile(corrected_valid, 2.5), np.percentile(corrected_valid, 97.5))
    
    # Determine recommended estimator
    if naive_rmse <= corrected_rmse:
        recommended = "Naive (uncorrected)"
    else:
        recommended = "Corrected"
    
    return SimulationResult(
        n_subjects=n_subjects,
        n_repeats=n_repeats,
        true_correlation=true_correlation,
        n_iterations=n_iterations,
        naive_mean=naive_mean,
        naive_sd=naive_sd,
        naive_rmse=naive_rmse,
        naive_ci=naive_ci,
        corrected_mean=corrected_mean,
        corrected_sd=corrected_sd,
        corrected_rmse=corrected_rmse,
        corrected_ci=corrected_ci,
        recommended_estimator=recommended
    )


def run_simulation_grid(
    task1_params: TaskParameters,
    task2_params: TaskParameters,
    sample_sizes: List[int] = [40, 80, 160, 320],
    n_repeats_list: List[int] = [2, 4, 8],
    true_correlation: float = 0.6,
    n_iterations: int = 500,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Run simulations across a grid of sample sizes and measurement counts.
    
    Useful for study planning to see how accuracy varies with design parameters.
    
    Parameters
    ----------
    task1_params, task2_params : TaskParameters
        Task parameters.
    sample_sizes : list of int
        Sample sizes to evaluate.
    n_repeats_list : list of int
        Numbers of repeated measurements to evaluate.
    true_correlation : float
        True correlation.
    n_iterations : int
        Iterations per condition.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Results table with one row per condition.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    results = []
    total_conditions = len(sample_sizes) * len(n_repeats_list)
    current = 0
    
    for n_subjects in sample_sizes:
        for n_repeats in n_repeats_list:
            current += 1
            print(f"Running condition {current}/{total_conditions}: N={n_subjects}, repeats={n_repeats}")
            
            sim_result = run_simulation(
                task1_params, task2_params,
                n_subjects=n_subjects,
                n_repeats=n_repeats,
                true_correlation=true_correlation,
                n_iterations=n_iterations
            )
            
            results.append({
                'n_subjects': n_subjects,
                'n_repeats': n_repeats,
                'true_correlation': true_correlation,
                'naive_mean': sim_result.naive_mean,
                'naive_sd': sim_result.naive_sd,
                'naive_rmse': sim_result.naive_rmse,
                'naive_ci_lower': sim_result.naive_ci[0],
                'naive_ci_upper': sim_result.naive_ci[1],
                'corrected_mean': sim_result.corrected_mean,
                'corrected_sd': sim_result.corrected_sd,
                'corrected_rmse': sim_result.corrected_rmse,
                'corrected_ci_lower': sim_result.corrected_ci[0],
                'corrected_ci_upper': sim_result.corrected_ci[1],
                'recommended': sim_result.recommended_estimator
            })
    
    return pd.DataFrame(results)


def get_simulation_distributions(
    task1_params: TaskParameters,
    task2_params: TaskParameters,
    n_subjects: int = 100,
    n_repeats: int = 4,
    true_correlation: float = 0.6,
    n_iterations: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Get full distributions of correlation estimates from simulation.
    
    Returns arrays of estimates for plotting histograms.
    
    Parameters
    ----------
    [Same as run_simulation]
    
    Returns
    -------
    dict
        Keys: 'naive', 'corrected'
        Values: Arrays of correlation estimates
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    naive_estimates = []
    corrected_estimates = []
    
    for _ in range(n_iterations):
        sim_data = simulate_test_retest_data(
            task1_params, task2_params,
            n_subjects=n_subjects,
            n_repeats=n_repeats,
            true_correlation=true_correlation
        )
        
        naive_r, _ = pearsonr(sim_data['task1_mean'], sim_data['task2_mean'])
        
        task1_cols = [f'task1_repeat{i+1}' for i in range(n_repeats)]
        task2_cols = [f'task2_repeat{i+1}' for i in range(n_repeats)]
        
        rel_task1 = compute_reliability(sim_data[task1_cols])
        rel_task2 = compute_reliability(sim_data[task2_cols])
        
        attenuation = np.sqrt(rel_task1 * rel_task2)
        corrected_r = naive_r / attenuation if attenuation > 0 else np.nan
        
        naive_estimates.append(naive_r)
        corrected_estimates.append(corrected_r)
    
    return {
        'naive': np.array(naive_estimates),
        'corrected': np.array(corrected_estimates)
    }


def estimate_required_sample_size(
    task1_params: TaskParameters,
    task2_params: TaskParameters,
    true_correlation: float = 0.6,
    n_repeats: int = 4,
    target_rmse: float = 0.1,
    sample_size_range: Tuple[int, int] = (20, 500),
    n_iterations: int = 500,
    random_state: Optional[int] = None
) -> Dict[str, int]:
    """
    Estimate required sample size to achieve a target RMSE.
    
    Uses binary search to find the minimum sample size needed.
    
    Parameters
    ----------
    task1_params, task2_params : TaskParameters
        Task parameters.
    true_correlation : float
        True correlation.
    n_repeats : int
        Number of repeated measurements.
    target_rmse : float
        Desired RMSE.
    sample_size_range : tuple
        (min, max) sample sizes to search.
    n_iterations : int
        Iterations per sample size.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    dict
        'naive': Required N for naive estimator
        'corrected': Required N for corrected estimator
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    def find_n_for_target(estimator_type: str) -> int:
        low, high = sample_size_range
        
        while high - low > 10:
            mid = (low + high) // 2
            
            result = run_simulation(
                task1_params, task2_params,
                n_subjects=mid,
                n_repeats=n_repeats,
                true_correlation=true_correlation,
                n_iterations=n_iterations
            )
            
            if estimator_type == 'naive':
                current_rmse = result.naive_rmse
            else:
                current_rmse = result.corrected_rmse
            
            if current_rmse <= target_rmse:
                high = mid
            else:
                low = mid
        
        return high
    
    return {
        'naive': find_n_for_target('naive'),
        'corrected': find_n_for_target('corrected')
    }


def power_analysis_correlation(
    effect_size: float,  # expected true correlation
    task1_reliability: float,
    task2_reliability: float,
    alpha: float = 0.05,
    power: float = 0.80,
    use_correction: bool = True
) -> int:
    """
    Estimate required sample size for detecting a correlation.
    
    Uses Fisher's z transformation for power calculation, accounting
    for attenuation if use_correction=False.
    
    Parameters
    ----------
    effect_size : float
        Expected true (latent) correlation.
    task1_reliability, task2_reliability : float
        Reliabilities of the two tasks.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    use_correction : bool
        If True, assumes corrected correlation will be used (full effect size).
        If False, assumes naive correlation (attenuated effect size).
    
    Returns
    -------
    int
        Required sample size.
    """
    from scipy.stats import norm
    
    # Attenuate effect size if not using correction
    if not use_correction:
        attenuation = np.sqrt(task1_reliability * task2_reliability)
        effect_size = effect_size * attenuation
    
    # Fisher's z transformation
    z_rho = 0.5 * np.log((1 + effect_size) / (1 - effect_size + 1e-10))
    
    # Critical values
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    # Sample size formula for correlation
    n = ((z_alpha + z_beta) / z_rho) ** 2 + 3
    
    return int(np.ceil(n))
