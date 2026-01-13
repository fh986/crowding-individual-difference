"""
Correlation Attenuation Toolbox - Visualization Module

This module provides plotting functions for:
1. Correlation heatmaps (naive vs. corrected)
2. Histograms of simulation/bootstrap distributions
3. RMSE vs. sample size plots for study planning
4. Scatter plots with reliability bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Union
import warnings


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_correlation_heatmaps(
    results_df: pd.DataFrame,
    task_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    cmap: str = 'RdBu_r',
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot side-by-side heatmaps of naive and corrected correlations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from analyze_all_pairs() with columns:
        task_x, task_y, naive_r, corrected_r
    task_names : list of str, optional
        Ordered list of task names for axes.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    vmin, vmax : float
        Color scale limits.
    annotate : bool
        Whether to show correlation values in cells.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    # Get unique tasks
    if task_names is None:
        task_names = sorted(set(results_df['task_x'].tolist() + results_df['task_y'].tolist()))
    
    n_tasks = len(task_names)
    task_idx = {name: i for i, name in enumerate(task_names)}
    
    # Create correlation matrices
    naive_matrix = np.eye(n_tasks)
    corrected_matrix = np.eye(n_tasks)
    
    for _, row in results_df.iterrows():
        i = task_idx[row['task_x']]
        j = task_idx[row['task_y']]
        naive_matrix[i, j] = naive_matrix[j, i] = row['naive_r']
        corrected_matrix[i, j] = corrected_matrix[j, i] = row['corrected_r']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Naive correlations
    im1 = axes[0].imshow(naive_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Naive (Uncorrected) Correlations', fontsize=14)
    axes[0].set_xticks(range(n_tasks))
    axes[0].set_yticks(range(n_tasks))
    axes[0].set_xticklabels(task_names, rotation=45, ha='right')
    axes[0].set_yticklabels(task_names)
    
    if annotate:
        for i in range(n_tasks):
            for j in range(n_tasks):
                text_color = 'white' if abs(naive_matrix[i, j]) > 0.5 else 'black'
                axes[0].text(j, i, f'{naive_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=10)
    
    # Corrected correlations
    im2 = axes[1].imshow(corrected_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Attenuation-Corrected Correlations', fontsize=14)
    axes[1].set_xticks(range(n_tasks))
    axes[1].set_yticks(range(n_tasks))
    axes[1].set_xticklabels(task_names, rotation=45, ha='right')
    axes[1].set_yticklabels(task_names)
    
    if annotate:
        for i in range(n_tasks):
            for j in range(n_tasks):
                val = corrected_matrix[i, j]
                # Handle values > 1 (can happen with correction)
                display_val = min(val, 1.0) if val > 0 else max(val, -1.0)
                text_color = 'white' if abs(display_val) > 0.5 else 'black'
                axes[1].text(j, i, f'{val:.2f}',
                           ha='center', va='center', color=text_color, fontsize=10)
    
    # Colorbar
    fig.colorbar(im2, ax=axes, shrink=0.8, label='Correlation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_correlation_histograms(
    naive_samples: np.ndarray,
    corrected_samples: np.ndarray,
    true_correlation: Optional[float] = None,
    title: str = 'Distribution of Correlation Estimates',
    figsize: Tuple[int, int] = (10, 4),
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histograms of naive and corrected correlation estimates.
    
    Parameters
    ----------
    naive_samples : np.ndarray
        Array of naive correlation estimates (from bootstrap or simulation).
    corrected_samples : np.ndarray
        Array of corrected correlation estimates.
    true_correlation : float, optional
        True correlation (for simulation) to show as reference line.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    bins : int
        Number of histogram bins.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Filter out NaN values
    corrected_valid = corrected_samples[~np.isnan(corrected_samples)]
    
    # Naive histogram
    axes[0].hist(naive_samples, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(naive_samples), color='red', linestyle='--', linewidth=2, label='Mean')
    if true_correlation is not None:
        axes[0].axvline(true_correlation, color='black', linestyle='-', linewidth=2, label=r'$\rho$')
    axes[0].set_xlabel('Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Naive (Uncorrected)')
    axes[0].legend()
    
    # Corrected histogram
    axes[1].hist(corrected_valid, bins=bins, alpha=0.7, color='darkorange', edgecolor='black')
    axes[1].axvline(np.mean(corrected_valid), color='red', linestyle='--', linewidth=2, label='Mean')
    if true_correlation is not None:
        axes[1].axvline(true_correlation, color='black', linestyle='-', linewidth=2, label=r'$\rho$')
    axes[1].set_xlabel('Correlation')
    axes[1].set_title('Attenuation-Corrected')
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_rmse_comparison(
    results_df: pd.DataFrame,
    x_variable: str = 'n_subjects',
    group_variable: str = 'n_repeats',
    true_correlation: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot RMSE of naive vs. corrected estimators across conditions.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_simulation_grid() with columns:
        n_subjects, n_repeats, naive_rmse, corrected_rmse
    x_variable : str
        Variable for x-axis (e.g., 'n_subjects' or 'n_repeats').
    group_variable : str
        Variable to group by (different lines).
    true_correlation : float, optional
        True correlation for reference.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    group_values = sorted(results_df[group_variable].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(group_values)))
    
    for i, group_val in enumerate(group_values):
        subset = results_df[results_df[group_variable] == group_val].sort_values(x_variable)
        
        x = subset[x_variable]
        
        ax.plot(x, subset['naive_rmse'], '-o', color=colors[i],
                label=f'{group_variable}={group_val} (Naive)')
        ax.plot(x, subset['corrected_rmse'], '--s', color=colors[i],
                label=f'{group_variable}={group_val} (Corrected)')
    
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel(x_variable.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_title('RMSE Comparison: Naive vs. Corrected Estimators', fontsize=14)
    
    if x_variable == 'n_subjects':
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_bias_variance_comparison(
    results_df: pd.DataFrame,
    x_variable: str = 'n_subjects',
    group_variable: str = 'n_repeats',
    true_correlation: float = 0.6,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot mean estimates and standard deviations for naive vs. corrected.
    
    Shows bias (deviation from true value) and variance (spread) of estimators.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_simulation_grid().
    x_variable : str
        Variable for x-axis.
    group_variable : str
        Variable to group by.
    true_correlation : float
        True correlation for reference.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    
    group_values = sorted(results_df[group_variable].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(group_values)))
    
    for i, group_val in enumerate(group_values):
        subset = results_df[results_df[group_variable] == group_val].sort_values(x_variable)
        x = subset[x_variable]
        
        # Mean estimates (bias visualization)
        axes[0].plot(x, subset['naive_mean'], '-o', color=colors[i],
                    label=f'{group_variable}={group_val} (Naive)')
        axes[0].plot(x, subset['corrected_mean'], '--s', color=colors[i],
                    label=f'{group_variable}={group_val} (Corrected)')
    
    axes[0].axhline(true_correlation, color='black', linestyle='-', linewidth=2, label='True r')
    axes[0].set_xlabel(x_variable.replace('_', ' ').title(), fontsize=14)
    axes[0].set_ylabel('Mean Correlation Estimate', fontsize=14)
    axes[0].set_title('Bias: Mean Estimates vs. True Correlation', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    for i, group_val in enumerate(group_values):
        subset = results_df[results_df[group_variable] == group_val].sort_values(x_variable)
        x = subset[x_variable]
        
        # Standard deviations (variance visualization)
        axes[1].plot(x, subset['naive_sd'], '-o', color=colors[i],
                    label=f'{group_variable}={group_val} (Naive)')
        axes[1].plot(x, subset['corrected_sd'], '--s', color=colors[i],
                    label=f'{group_variable}={group_val} (Corrected)')
    
    axes[1].set_xlabel(x_variable.replace('_', ' ').title(), fontsize=14)
    axes[1].set_ylabel('Standard Deviation', fontsize=14)
    axes[1].set_title('Variance: SD of Correlation Estimates', fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    if x_variable == 'n_subjects':
        for ax in axes:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_scatter_with_reliability(
    x: np.ndarray,
    y: np.ndarray,
    reliability_x: float,
    reliability_y: float,
    naive_r: float,
    corrected_r: float,
    xlabel: str = 'Task 1',
    ylabel: str = 'Task 2',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot scatter plot with regression lines for naive and corrected correlations.
    
    Parameters
    ----------
    x, y : np.ndarray
        Data arrays (subject means).
    reliability_x, reliability_y : float
        Reliabilities of each variable.
    naive_r, corrected_r : float
        Correlation estimates.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='white')
    
    # Regression lines
    x_range = np.linspace(x.min(), x.max(), 100)
    
    # Naive regression line
    slope_naive = naive_r * np.std(y) / np.std(x)
    intercept_naive = np.mean(y) - slope_naive * np.mean(x)
    ax.plot(x_range, slope_naive * x_range + intercept_naive, 
            'b-', linewidth=2, label=f'Naive r = {naive_r:.3f}')
    
    # Corrected regression line (steeper if corrected_r > naive_r)
    slope_corrected = corrected_r * np.std(y) / np.std(x)
    intercept_corrected = np.mean(y) - slope_corrected * np.mean(x)
    ax.plot(x_range, slope_corrected * x_range + intercept_corrected,
            'r--', linewidth=2, label=f'Corrected r = {corrected_r:.3f}')
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'Correlation: Naive vs. Corrected\n'
                f'(Reliability X = {reliability_x:.3f}, Y = {reliability_y:.3f})',
                fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ratio = np.var(y) / np.var(x)
    ax.set_aspect(ratio, adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_simulation_summary(
    results_df: pd.DataFrame,
    true_correlation: float = 0.6,
    figsize: Tuple[int, int] = (14, 12), # Slightly wider to accommodate legend
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive summary plot of simulation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    sample_sizes = sorted(results_df['n_subjects'].unique())
    n_repeats_list = sorted(results_df['n_repeats'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_repeats_list)))
    
    # Top left: Mean estimates
    ax = axes[0, 0]
    for i, n_rep in enumerate(n_repeats_list):
        subset = results_df[results_df['n_repeats'] == n_rep].sort_values('n_subjects')
        ax.plot(subset['n_subjects'], subset['naive_mean'], '-o', color=colors[i],
                label=f'{n_rep} repeats (Naive)')
        ax.plot(subset['n_subjects'], subset['corrected_mean'], '--s', color=colors[i],
                label=f'{n_rep} repeats (Corrected)')
    ax.axhline(true_correlation, color='black', linestyle='-', linewidth=2, label=r'$\rho$')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Estimate')
    ax.set_title('Mean Correlation Estimates')
    
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.minorticks_off()
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    
    # Capture handles and labels for the single legend from the first plot
    handles, labels = ax.get_legend_handles_labels()
    
    # Top right: RMSE
    ax = axes[0, 1]
    for i, n_rep in enumerate(n_repeats_list):
        subset = results_df[results_df['n_repeats'] == n_rep].sort_values('n_subjects')
        ax.plot(subset['n_subjects'], subset['naive_rmse'], '-o', color=colors[i])
        ax.plot(subset['n_subjects'], subset['corrected_rmse'], '--s', color=colors[i])
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error')
    
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.minorticks_off()
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Standard deviation
    ax = axes[1, 0]
    for i, n_rep in enumerate(n_repeats_list):
        subset = results_df[results_df['n_repeats'] == n_rep].sort_values('n_subjects')
        ax.plot(subset['n_subjects'], subset['naive_sd'], '-o', color=colors[i])
        ax.plot(subset['n_subjects'], subset['corrected_sd'], '--s', color=colors[i])
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Sampling Variability')
    
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    ax.minorticks_off()
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Recommended estimator
    ax = axes[1, 1]
    pivot = results_df.pivot(index='n_subjects', columns='n_repeats', values='recommended')
    pivot_numeric = pivot.applymap(lambda x: 1 if 'Corrected' in str(x) else 0)
    
    im = ax.imshow(pivot_numeric.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(n_repeats_list)))
    ax.set_xticklabels(n_repeats_list)
    ax.set_yticks(range(len(sample_sizes)))
    ax.set_yticklabels(sample_sizes)
    ax.set_xlabel('Number of Repeats')
    ax.set_ylabel('Sample Size')
    ax.set_title('Recommended Estimator\n(Red=Naive, Green=Corrected)')
    
    ax.set_box_aspect(1)
    
    for i in range(len(sample_sizes)):
        for j in range(len(n_repeats_list)):
            val = pivot_numeric.iloc[i, j]
            text = 'C' if val == 1 else 'N'
            color = 'white' if val == 1 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12, fontweight='bold')
    
    # Main Title with Rho
    plt.suptitle(rf'Simulation Summary ($\rho = {true_correlation}$)', fontsize=16)
    
    # Place single legend to the right
    # bbox_to_anchor centers the legend vertically (0.5) and places it just outside the right edge (1.05)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=10)
    
    # Adjust layout to make room for legend on the right
    # rect=[left, bottom, right, top] in normalized (0, 1) figure coordinates
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_reliability_effect(
    reliabilities: np.ndarray = np.linspace(0.5, 1.0, 20),
    observed_correlations: List[float] = [0.3, 0.5, 0.7],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how corrected correlation varies with reliability.
    
    Shows the relationship between reliability, observed correlation,
    and corrected correlation.
    
    Parameters
    ----------
    reliabilities : np.ndarray
        Range of reliability values to plot.
    observed_correlations : list of float
        Different observed correlations to show.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(observed_correlations)))
    
    for obs_r, color in zip(observed_correlations, colors):
        # Assuming equal reliability for both variables
        attenuation = reliabilities
        corrected = obs_r / attenuation
        
        ax.plot(reliabilities, corrected, '-', color=color, linewidth=2,
                label=f'Observed r = {obs_r}')
        ax.axhline(obs_r, color=color, linestyle=':', alpha=0.5)
    
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='r = 1.0')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Reliability (equal for both variables)', fontsize=14)
    ax.set_ylabel('Corrected Correlation', fontsize=14)
    ax.set_title('Effect of Reliability on Corrected Correlation', fontsize=14)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
