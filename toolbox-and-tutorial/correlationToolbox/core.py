"""
Correlation Attenuation Toolbox - Core Module

This module provides functions for correcting correlation coefficients
for attenuation due to measurement noise.

The correction is based on the classical formula:
    r_corrected = r_observed / sqrt(reliability_x * reliability_y)

where reliability is estimated from test-retest measurements.

Reference: Spearman, C. (1904). The proof and measurement of association 
between two things. American Journal of Psychology, 15, 72-101.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class TaskParameters:
    """
    Parameters for a single task/variable.
    
    Attributes:
        between_var: Between-subject variance (true individual differences)
        within_var: Within-subject variance (measurement noise)
        mean: Mean of the variable
        name: Optional name for the task
    """
    between_var: float
    within_var: float
    mean: float = 0.0
    name: str = "task"
    
    @property
    def reliability(self) -> float:
        """Compute theoretical reliability from variances."""
        total_var = self.between_var + self.within_var
        if total_var <= 0:
            return np.nan
        return self.between_var / total_var


@dataclass
class CorrelationResult:
    """
    Results from correlation analysis.
    
    Attributes:
        naive_r: Uncorrected (naive) correlation coefficient
        corrected_r: Attenuation-corrected correlation coefficient
        reliability_x: Reliability estimate for variable X
        reliability_y: Reliability estimate for variable Y
        attenuation_factor: sqrt(reliability_x * reliability_y)
        naive_ci: 95% CI for naive correlation [lower, upper]
        corrected_ci: 95% CI for corrected correlation [lower, upper]
        n_subjects: Number of subjects in the analysis
        n_repeats_x: Number of repeated measurements for X
        n_repeats_y: Number of repeated measurements for Y
    """
    naive_r: float
    corrected_r: float
    reliability_x: float
    reliability_y: float
    attenuation_factor: float
    naive_ci: Optional[Tuple[float, float]] = None
    corrected_ci: Optional[Tuple[float, float]] = None
    n_subjects: int = 0
    n_repeats_x: int = 0
    n_repeats_y: int = 0
    
    def __str__(self):
        ci_naive = f"[{self.naive_ci[0]:.3f}, {self.naive_ci[1]:.3f}]" if self.naive_ci else "N/A"
        ci_corr = f"[{self.corrected_ci[0]:.3f}, {self.corrected_ci[1]:.3f}]" if self.corrected_ci else "N/A"
        return (
            f"Correlation Analysis Results (N = {self.n_subjects})\n"
            f"{'='*50}\n"
            f"r:               {self.naive_r:.3f}  {ci_naive}\n"
            f"r_unbiased:      {self.corrected_r:.3f}  {ci_corr}\n"
            f"Reliability X:   {self.reliability_x:.3f}\n"
            f"Reliability Y:   {self.reliability_y:.3f}\n"
            f"r_ceiling:       {self.attenuation_factor:.3f}\n"
        )

def compute_variances(measurements: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute within-subject and between-subject variances from repeated measurements.
    
    Parameters
    ----------
    measurements : pd.DataFrame
        DataFrame of shape (n_subjects, n_repeats), where each row is a subject
        and each column is a repeated measurement.
    
    Returns
    -------
    within_var : float
        Average within-subject variance (measurement noise).
    between_var : float
        Between-subject variance of the means (individual differences).
    """
    # Within-subject variance: average variance across repeats within each subject
    within_var = measurements.var(axis=1, ddof=1, skipna=True).mean(skipna=True)
    
    # Between-subject variance: variance of subject means
    subject_means = measurements.mean(axis=1, skipna=True)
    between_var = subject_means.var(ddof=1, skipna=True)
    
    return within_var, between_var


def compute_reliability(measurements: pd.DataFrame) -> float:
    """
    Compute reliability of the mean of k repeated measurements.
    
    Uses the formula: R = Var_between / (Var_between + Var_within/k)
    
    This is equivalent to Cronbach's alpha for parallel tests and represents
    the proportion of variance in the observed mean that is due to true
    individual differences rather than measurement error.
    
    Parameters
    ----------
    measurements : pd.DataFrame
        DataFrame of shape (n_subjects, n_repeats).
    
    Returns
    -------
    reliability : float
        Reliability coefficient (0 to 1, can exceed 1 in edge cases).
    """
    k = measurements.shape[1]
    if k < 2:
        raise ValueError("Need at least 2 repeated measurements to compute reliability.")
    
    means = measurements.mean(axis=1, skipna=True)
    between_var = means.var(ddof=1, skipna=True)
    within_var = measurements.var(axis=1, ddof=1, skipna=True).mean(skipna=True)
    
    denominator = between_var + within_var / k
    if denominator <= 0 or np.isnan(denominator):
        return np.nan
    
    return between_var / denominator


def compute_reliability_split_half(measurements: pd.DataFrame) -> float:
    """
    Compute reliability using split-half method with Spearman-Brown correction.
    
    Splits measurements into odd and even halves, correlates their means,
    and applies the Spearman-Brown prophecy formula.
    
    Parameters
    ----------
    measurements : pd.DataFrame
        DataFrame of shape (n_subjects, n_repeats).
    
    Returns
    -------
    reliability : float
        Split-half reliability with Spearman-Brown correction.
    """
    n_measurements = measurements.shape[1]
    if n_measurements < 2:
        raise ValueError("Need at least 2 repeated measurements.")
    
    # Split into odd and even indexed columns
    odd_half = measurements.iloc[:, 1::2]
    even_half = measurements.iloc[:, 0::2]
    
    # Compute means of each half
    mean_odd = odd_half.mean(axis=1, skipna=True)
    mean_even = even_half.mean(axis=1, skipna=True)
    
    # Mask NaN values for correlation
    mask = mean_odd.notna() & mean_even.notna()
    if mask.sum() < 3:
        return np.nan
    
    # Correlate the halves
    r_half, _ = pearsonr(mean_odd[mask], mean_even[mask])
    
    # Spearman-Brown correction (for 2 halves -> full test)
    r_full = (2 * r_half) / (1 + r_half)
    
    return r_full


def correct_correlation(r_observed: float, 
                       reliability_x: float, 
                       reliability_y: float) -> float:
    """
    Correct an observed correlation for attenuation due to measurement unreliability.
    
    Uses Spearman's classical correction formula:
        r_true = r_observed / sqrt(reliability_x * reliability_y)
    
    Parameters
    ----------
    r_observed : float
        Observed (uncorrected) correlation coefficient.
    reliability_x : float
        Reliability of variable X.
    reliability_y : float
        Reliability of variable Y.
    
    Returns
    -------
    r_corrected : float
        Corrected correlation estimate. May exceed 1.0 in small samples.
    """
    if np.isnan(reliability_x) or np.isnan(reliability_y):
        return np.nan
    attenuation = np.sqrt(reliability_x * reliability_y)
    if attenuation <= 0:
        return np.nan
    return r_observed / attenuation


def analyze_correlation(task1_measurements: pd.DataFrame,
                       task2_measurements: pd.DataFrame,
                       n_bootstrap: int = 1000,
                       confidence_level: float = 0.95,
                       random_state: Optional[int] = None) -> CorrelationResult:
    """
    Perform complete correlation analysis with attenuation correction and CIs.
    
    This is the main function for analyzing existing data. It computes both
    naive and corrected correlations along with bootstrap confidence intervals.
    
    Parameters
    ----------
    task1_measurements : pd.DataFrame
        DataFrame of shape (n_subjects, n_repeats) for task 1.
        Each row is a subject, each column is a repeated measurement.
    task2_measurements : pd.DataFrame
        DataFrame of shape (n_subjects, n_repeats) for task 2.
        Must have the same number of rows as task1_measurements.
    n_bootstrap : int, default=1000
        Number of bootstrap iterations for confidence intervals.
    confidence_level : float, default=0.95
        Confidence level for intervals.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    CorrelationResult
        Object containing naive r, corrected r, reliabilities, and CIs.
    
    Example
    -------
    >>> import pandas as pd
    >>> # Task 1: 50 subjects, 4 repeats
    >>> task1 = pd.DataFrame(np.random.randn(50, 4))
    >>> # Task 2: 50 subjects, 4 repeats
    >>> task2 = pd.DataFrame(np.random.randn(50, 4))
    >>> result = analyze_correlation(task1, task2)
    >>> print(result)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_subjects = len(task1_measurements)
    if len(task2_measurements) != n_subjects:
        raise ValueError("Task measurements must have the same number of subjects.")
    
    # Compute subject means for correlation
    means_x = task1_measurements.mean(axis=1, skipna=True).values
    means_y = task2_measurements.mean(axis=1, skipna=True).values
    
    # Mask for valid (non-NaN) pairs
    valid_mask = ~np.isnan(means_x) & ~np.isnan(means_y)
    n_valid = valid_mask.sum()
    
    if n_valid < 3:
        raise ValueError("Need at least 3 valid subjects with data for both tasks.")
    
    # Point estimates (using only valid pairs)
    naive_r, _ = pearsonr(means_x[valid_mask], means_y[valid_mask])
    rel_x = compute_reliability(task1_measurements[valid_mask])
    rel_y = compute_reliability(task2_measurements[valid_mask])
    attenuation = np.sqrt(rel_x * rel_y) if not (np.isnan(rel_x) or np.isnan(rel_y)) else np.nan
    corrected_r = naive_r / attenuation if attenuation and attenuation > 0 else np.nan
    
    # Bootstrap for confidence intervals
    naive_samples = []
    corrected_samples = []
    
    # Get indices of valid subjects for bootstrapping
    valid_indices = np.where(valid_mask)[0]
    
    for _ in range(n_bootstrap):
        # Resample valid subjects with replacement
        idx = np.random.choice(valid_indices, size=len(valid_indices), replace=True)
        
        boot_task1 = task1_measurements.iloc[idx].reset_index(drop=True)
        boot_task2 = task2_measurements.iloc[idx].reset_index(drop=True)
        
        boot_means_x = boot_task1.mean(axis=1, skipna=True).values
        boot_means_y = boot_task2.mean(axis=1, skipna=True).values
        
        # Mask for this bootstrap sample
        boot_mask = ~np.isnan(boot_means_x) & ~np.isnan(boot_means_y)
        if boot_mask.sum() < 3:
            naive_samples.append(np.nan)
            corrected_samples.append(np.nan)
            continue
        
        boot_naive_r, _ = pearsonr(boot_means_x[boot_mask], boot_means_y[boot_mask])
        boot_rel_x = compute_reliability(boot_task1[boot_mask])
        boot_rel_y = compute_reliability(boot_task2[boot_mask])
        boot_atten = np.sqrt(boot_rel_x * boot_rel_y) if not (np.isnan(boot_rel_x) or np.isnan(boot_rel_y)) else np.nan
        boot_corrected_r = boot_naive_r / boot_atten if boot_atten and boot_atten > 0 else np.nan
        
        naive_samples.append(boot_naive_r)
        corrected_samples.append(boot_corrected_r)
    
    # Compute CIs using percentile method
    alpha = 1 - confidence_level
    naive_ci = (
        np.nanpercentile(naive_samples, 100 * alpha / 2),
        np.nanpercentile(naive_samples, 100 * (1 - alpha / 2))
    )
    corrected_ci = (
        np.nanpercentile(corrected_samples, 100 * alpha / 2),
        np.nanpercentile(corrected_samples, 100 * (1 - alpha / 2))
    )
    
    return CorrelationResult(
        naive_r=naive_r,
        corrected_r=corrected_r,
        reliability_x=rel_x,
        reliability_y=rel_y,
        attenuation_factor=attenuation,
        naive_ci=naive_ci,
        corrected_ci=corrected_ci,
        n_subjects=n_valid,  # Report number of valid subjects
        n_repeats_x=task1_measurements.shape[1],
        n_repeats_y=task2_measurements.shape[1]
    )


def analyze_all_pairs(data: pd.DataFrame,
                     task_prefixes: List[str],
                     n_bootstrap: int = 1000,
                     confidence_level: float = 0.95,
                     random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Analyze correlations between all pairs of tasks in a dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where columns are named like "task1_repeat1", "task1_repeat2", etc.
    task_prefixes : list of str
        List of task name prefixes (e.g., ["task1", "task2", "task3"]).
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    confidence_level : float, default=0.95
        Confidence level for intervals.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Results table with columns for each task pair's correlations and CIs.
    """
    results = []
    
    for i, task_x in enumerate(task_prefixes):
        for task_y in task_prefixes[i+1:]:
            # Extract measurements for each task
            cols_x = [c for c in data.columns if c.startswith(task_x)]
            cols_y = [c for c in data.columns if c.startswith(task_y)]
            
            if len(cols_x) < 2 or len(cols_y) < 2:
                continue
            
            task1_data = data[cols_x]
            task2_data = data[cols_y]
            
            # Find subjects with at least some data in both tasks
            has_x = task1_data.notna().any(axis=1)
            has_y = task2_data.notna().any(axis=1)
            common_idx = data.index[has_x & has_y]
            
            task1_data = task1_data.loc[common_idx]
            task2_data = task2_data.loc[common_idx]
            
            if len(task1_data) < 10:
                continue
            
            try:
                result = analyze_correlation(
                    task1_data, task2_data,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    random_state=random_state
                )
                
                results.append({
                    'task_x': task_x,
                    'task_y': task_y,
                    'n_subjects': result.n_subjects,
                    'naive_r': result.naive_r,
                    'naive_ci_lower': result.naive_ci[0],
                    'naive_ci_upper': result.naive_ci[1],
                    'corrected_r': result.corrected_r,
                    'corrected_ci_lower': result.corrected_ci[0],
                    'corrected_ci_upper': result.corrected_ci[1],
                    'reliability_x': result.reliability_x,
                    'reliability_y': result.reliability_y,
                    'attenuation_factor': result.attenuation_factor
                })
            except ValueError:
                # Skip pairs with insufficient valid data
                continue
    
    return pd.DataFrame(results)


def extract_task_measurements(data: pd.DataFrame, 
                             task_prefix: str) -> pd.DataFrame:
    """
    Extract all repeated measurements for a task from a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with columns like "task1_repeat1", "task1_repeat2", etc.
    task_prefix : str
        Prefix for the task (e.g., "task1").
    
    Returns
    -------
    pd.DataFrame
        Subset with only columns matching the task prefix.
    """
    cols = [c for c in data.columns if c.startswith(task_prefix)]
    return data[cols]


def summarize_task_statistics(data: pd.DataFrame,
                             task_prefixes: List[str]) -> pd.DataFrame:
    """
    Compute summary statistics for each task.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with task measurements.
    task_prefixes : list of str
        List of task name prefixes.
    
    Returns
    -------
    pd.DataFrame
        Summary table with mean, within-var, between-var, reliability for each task.
    """
    summaries = []
    
    for task in task_prefixes:
        cols = [c for c in data.columns if c.startswith(task)]
        if len(cols) < 2:
            continue
        
        measurements = data[cols].dropna()
        within_var, between_var = compute_variances(measurements)
        reliability = compute_reliability(measurements)
        
        subject_means = measurements.mean(axis=1)
        
        summaries.append({
            'task': task,
            'n_subjects': len(measurements),
            'n_repeats': len(cols),
            'mean': subject_means.mean(),
            'within_subject_var': within_var,
            'between_subject_var': between_var,
            'reliability': reliability
        })
    
    return pd.DataFrame(summaries)
