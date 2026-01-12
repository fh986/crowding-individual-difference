# Bias-Corrected Correlation Toolbox

A Python toolkit for correcting correlation coefficients for attenuation due to measurement noise.

## Overview

When measuring the correlation between two variables, measurement noise causes the observed correlation to **underestimate** the true correlation. This is called **attenuation**. 

Correcting for attenuation (by dividing $r$ by the joint reliability of $X$ and $Y$) **removes bias** but **increases variance**. Whether correction improves accuracy depends on the sample size, measurement reliability, and true underlying correlation. 

This toolbox computes naive and corrected correlations with confidence intervals, and it specifies which correlation estimator (naive or corrected) has lower total error under your specified circumstances.

## Installation

Use the following code to install the toolbox, then refer to the **tutorial** notebook `tutorial.ipynb` for detailed usage examples.

```bash
# Clone the repository
git clone https://github.com/your-repo/attenuation-toolbox.git
cd attenuation-toolbox

# Install dependencies
pip install numpy pandas scipy matplotlib

# Or install as a package
pip install -e .
```

## Example function and output

```python
from attenuation_toolbox import analyze_correlation
import pandas as pd

# Your data: subjects × repeated measurements for each task
task1_data = pd.DataFrame({
    'task1_repeat1': [...],
    'task1_repeat2': [...],
    'task1_repeat3': [...],
})

task2_data = pd.DataFrame({
    'task2_repeat1': [...],
    'task2_repeat2': [...],
    'task2_repeat3': [...],
})

# Analyze with bootstrap CIs
result = analyze_correlation(task1_data, task2_data, n_bootstrap=2000)
print(result)
```

Output:
```
Correlation Analysis Results (N=100)
==================================================
Naive (uncorrected) r:     0.4823  95% CI: [0.321, 0.618]
Corrected r:               0.5891  95% CI: [0.392, 0.756]
Reliability X:             0.872
Reliability Y:             0.894
Attenuation factor:        0.883
```

## Data Format

Your data should be organized as a pandas DataFrame where:
- Each **row** is a subject/participant
- Each **column** is a measurement
- Column names follow the pattern: `taskname_repeatN`

Example:
```
| subject | task1_repeat1 | task1_repeat2 | task2_repeat1 | task2_repeat2 |
|---------|---------------|---------------|---------------|---------------|
| 1       | 0.52          | 0.48          | 1.20          | 1.35          |
| 2       | 0.31          | 0.29          | 0.95          | 0.88          |
| 3       | 0.45          | 0.51          | 1.15          | 1.22          |
```

## Key Functions

### Analysis (for existing data)

| Function | Description |
|----------|-------------|
| `analyze_correlation()` | Analyze correlation between two tasks with bootstrap CIs |
| `analyze_all_pairs()` | Analyze all pairwise correlations for multiple tasks |
| `compute_reliability()` | Compute reliability from repeated measurements |
| `summarize_task_statistics()` | Get summary stats for all tasks |

### Simulation (for study planning)

| Function | Description |
|----------|-------------|
| `simulate_test_retest_data()` | Generate synthetic data with known parameters |
| `run_simulation()` | Monte Carlo simulation for one condition |
| `run_simulation_grid()` | Evaluate multiple sample sizes and repeat counts |
| `power_analysis_correlation()` | Estimate required sample size |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_correlation_heatmaps()` | Side-by-side naive vs. corrected heatmaps |
| `plot_correlation_histograms()` | Bootstrap/simulation distributions |
| `plot_rmse_comparison()` | RMSE vs. sample size plots |
| `plot_simulation_summary()` | Comprehensive 4-panel summary |

## The Bias-Correction Formula

The classical correction (Spearman, 1904):

$$r_{corrected} = \frac{r_{observed}}{\sqrt{\text{reliability}_X \times \text{reliability}_Y}}$$

Where reliability is estimated as:

$$\text{reliability} = \frac{\sigma^2_{between}}{\sigma^2_{between} + \sigma^2_{within}/k}$$

- $\sigma^2_{between}$: Between-subject variance (true individual differences)
- $\sigma^2_{within}$: Within-subject variance (measurement noise)
- $k$: Number of repeated measurements


## Requirements

- Python 3.7+
- numpy
- pandas
- scipy
- matplotlib

## References

- Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15, 72-101.

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
