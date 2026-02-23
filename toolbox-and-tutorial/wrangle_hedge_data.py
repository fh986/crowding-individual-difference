"""
Wrangle Hedge, Powell, & Sumner (2018) data for the attenuation-correction toolbox.

This script:
  1. Loads trial-level CSV files for the Flanker and Stroop tasks
  2. Filters to correct trials and trims extreme RTs
  3. Splits each participant's trials into K repeats
  4. Computes interference effects (incongruent - congruent mean RT) per repeat
  5. Outputs a wide-format DataFrame ready for the toolbox

Expected folder structure:
    data_root/
        Flanker/
            Study1_P1Flanker1.csv
            Study1_P1Flanker2.csv
            Study2_P1Flanker1.csv
            ...
        Stroop/
            Study1_P1Stroop1.csv
            ...

CSV columns (no header):
    Flanker: block, trial, arrow_dir, condition, correct, rt
    Stroop:  block, trial, unused, condition, correct, rt
    condition: 0=congruent, 1=neutral, 2=incongruent

Usage:
    df = build_toolbox_dataframe(data_root, n_repeats=10)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
import warnings


# ---------------------------------------------------------------------------
# 1. Loading raw data
# ---------------------------------------------------------------------------

COLUMN_NAMES = ['block', 'trial', 'col3', 'condition', 'correct', 'rt']


def load_single_csv(filepath):
    """Load one CSV file and return a DataFrame with standard column names."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    return df


def parse_filename(filename):
    """
    Extract participant ID, study, and session from a filename.

    Examples:
        'Study1_P1Flanker1.csv'  -> participant='Study1_P1', session=1
        'Study2_P12Flanker2.csv' -> participant='Study2_P12', session=2
        'Study1_P3Stroop1.csv'   -> participant='Study1_P3', session=1
    """
    stem = Path(filename).stem  # e.g. 'Study1_P1Flanker1'

    # Match pattern: (Study\d+_P\d+)(Flanker|Stroop)(\d+)
    match = re.match(r'(Study\d+_P\d+)(Flanker|Stroop)(\d+)', stem)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    participant = match.group(1)
    task = match.group(2)
    session = int(match.group(3))

    return participant, task, session


def load_task_folder(folder_path):
    """
    Load all CSV files from a single task folder (e.g., 'Flanker/').

    Returns a long-format DataFrame with columns:
        participant, session, block, trial, condition, correct, rt
    """
    folder = Path(folder_path)
    all_dfs = []

    for csv_file in sorted(folder.glob('*.csv')):
        participant, task, session = parse_filename(csv_file.name)
        df = load_single_csv(csv_file)
        df['participant'] = participant
        df['session'] = session
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Create a global trial index per participant (across sessions and blocks)
    # so we can split trials in various ways later
    combined = combined.sort_values(['participant', 'session', 'block', 'trial'])
    combined['global_trial'] = combined.groupby('participant').cumcount()

    print(f"  Loaded {len(all_dfs)} files from {folder.name}/")
    print(f"  {combined['participant'].nunique()} unique participants")
    print(f"  {len(combined)} total trials")

    return combined


# ---------------------------------------------------------------------------
# 2. Preprocessing: filter and trim
# ---------------------------------------------------------------------------

def preprocess_trials(df, rt_min=0.2, rt_max_sd=3.0, use_conditions=(0, 2)):
    """
    Filter and trim trial-level data.

    Steps:
        1. Keep only congruent (0) and incongruent (2) trials by default
        2. Keep only correct trials
        3. Trim RTs below rt_min (seconds)
        4. Trim RTs beyond rt_max_sd standard deviations from each
           participant's mean (computed within the kept conditions)

    Parameters
    ----------
    df : DataFrame
        Long-format trial data with columns: participant, condition, correct, rt
    rt_min : float
        Minimum RT in seconds (default 0.2)
    rt_max_sd : float
        Number of SDs for per-participant trimming (default 3.0)
    use_conditions : tuple
        Which condition codes to keep (default (0, 2) = congruent & incongruent)

    Returns
    -------
    DataFrame with same columns, filtered rows
    """
    n_start = len(df)

    # Step 1: keep relevant conditions
    df = df[df['condition'].isin(use_conditions)].copy()
    n_after_cond = len(df)

    # Step 2: keep correct trials only
    df = df[df['correct'] == 1].copy()
    n_after_correct = len(df)

    # Step 3: trim very fast RTs
    df = df[df['rt'] >= rt_min].copy()
    n_after_fast = len(df)

    # Step 4: per-participant SD trimming
    stats = df.groupby('participant')['rt'].agg(['mean', 'std'])
    df = df.merge(stats, on='participant', suffixes=('', '_stats'))
    df = df[
        (df['rt'] >= df['mean'] - rt_max_sd * df['std']) &
        (df['rt'] <= df['mean'] + rt_max_sd * df['std'])
    ].copy()
    df = df.drop(columns=['mean', 'std'])
    n_after_sd = len(df)

    print(f"  Preprocessing: {n_start} -> {n_after_cond} (conditions) "
          f"-> {n_after_correct} (correct) -> {n_after_fast} (fast RT) "
          f"-> {n_after_sd} (SD trim)")

    return df


# ---------------------------------------------------------------------------
# 3. Splitting trials into repeats
# ---------------------------------------------------------------------------

def assign_repeats(df, n_repeats=10, method='interleaved'):
    """
    Assign each trial to one of n_repeats groups.

    Parameters
    ----------
    df : DataFrame
        Must have 'participant' and 'global_trial' columns.
    n_repeats : int
        Number of repeat groups to create.
    method : str
        'interleaved' - assign trials round-robin by global order (like odd/even
                        but generalized to K groups). This balances across time.
        'blocked'     - split trials into contiguous blocks.
        'random'      - random assignment (set a seed externally for reproducibility).

    Returns
    -------
    DataFrame with an added 'repeat' column (1-indexed: 1, 2, ..., n_repeats)
    """
    df = df.copy()

    if method == 'interleaved':
        # Within each participant, assign round-robin by trial order
        df['repeat'] = df.groupby('participant').cumcount() % n_repeats + 1

    elif method == 'blocked':
        # Within each participant, split into contiguous chunks
        def assign_blocks(group):
            n = len(group)
            return np.repeat(np.arange(1, n_repeats + 1),
                             np.diff(np.round(np.linspace(0, n, n_repeats + 1)).astype(int)))[:n]
        df['repeat'] = df.groupby('participant', group_keys=False).apply(
            lambda g: pd.Series(assign_blocks(g), index=g.index)
        )

    elif method == 'random':
        def assign_random(group):
            n = len(group)
            assignments = np.tile(np.arange(1, n_repeats + 1), n // n_repeats + 1)[:n]
            np.random.shuffle(assignments)
            return assignments
        df['repeat'] = df.groupby('participant', group_keys=False).apply(
            lambda g: pd.Series(assign_random(g), index=g.index)
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return df


# ---------------------------------------------------------------------------
# 4. Computing interference effects per repeat
# ---------------------------------------------------------------------------

def compute_interference_effects(df, task_name):
    """
    Compute the interference effect (mean RT incongruent - mean RT congruent)
    for each participant and repeat.

    Parameters
    ----------
    df : DataFrame
        Trial-level data with columns: participant, condition, rt, repeat
    task_name : str
        Name prefix for the output columns (e.g., 'flanker', 'stroop')

    Returns
    -------
    DataFrame with columns: participant, {task_name}_repeat1, ..., {task_name}_repeatK
    """
    # Mean RT per participant × repeat × condition
    means = (df.groupby(['participant', 'repeat', 'condition'])['rt']
             .mean()
             .unstack('condition'))

    # Interference = incongruent (2) minus congruent (0)
    effects = (means[2] - means[0]).reset_index()
    effects.columns = ['participant', 'repeat', 'effect']

    # Pivot to wide format: one column per repeat
    wide = effects.pivot(index='participant', columns='repeat', values='effect')
    wide.columns = [f'{task_name}_repeat{int(c)}' for c in wide.columns]
    wide = wide.reset_index()

    return wide


def compute_overall_effect(df, task_name):
    """
    Compute the overall interference effect using ALL trials (no splitting).
    This serves as the participant's single best estimate.

    Returns
    -------
    DataFrame with columns: participant, {task_name}_overall
    """
    means = (df.groupby(['participant', 'condition'])['rt']
             .mean()
             .unstack('condition'))

    effects = (means[2] - means[0]).reset_index()
    effects.columns = ['participant', f'{task_name}_overall']

    return effects


# ---------------------------------------------------------------------------
# 5. Main assembly function
# ---------------------------------------------------------------------------

def build_toolbox_dataframe(data_root, n_repeats=10, split_method='interleaved',
                            rt_min=0.2, rt_max_sd=3.0, subsample_fraction=None,
                            random_state=None):
    """
    Full pipeline: load -> preprocess -> split -> compute effects -> merge.

    Parameters
    ----------
    data_root : str or Path
        Root directory containing 'Flanker/' and 'Stroop/' subfolders.
    n_repeats : int
        Number of trial splits per participant (used as "repeats" by the toolbox).
    split_method : str
        How to assign trials to repeats: 'interleaved', 'blocked', or 'random'.
    rt_min : float
        Minimum RT threshold in seconds.
    rt_max_sd : float
        Per-participant SD trimming threshold.
    subsample_fraction : float or None
        If set (e.g., 0.1), randomly subsample this fraction of each participant's
        trials BEFORE splitting into repeats. Useful for the ground-truth comparison
        demo: compute effects from a subset, compare to full-data effects.
    random_state : int or None
        Random seed (used for 'random' split method and for subsampling).

    Returns
    -------
    DataFrame ready for the toolbox, with columns:
        participant,
        flanker_repeat1, ..., flanker_repeatK,
        stroop_repeat1, ..., stroop_repeatK,
        flanker_overall, stroop_overall
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_root = Path(data_root)

    results = {}
    for task_name, folder_name in [('flanker', 'Flanker'), ('stroop', 'Stroop')]:
        folder_path = data_root / folder_name
        print(f"\n--- {task_name.upper()} ---")

        # Load
        raw = load_task_folder(folder_path)

        # Preprocess
        clean = preprocess_trials(raw, rt_min=rt_min, rt_max_sd=rt_max_sd)

        # Compute overall effect BEFORE any subsampling (this is the "ground truth")
        overall = compute_overall_effect(clean, task_name)

        # Optional subsampling
        if subsample_fraction is not None:
            print(f"  Subsampling {subsample_fraction:.0%} of trials per participant...")
            sampled_idx = (clean.groupby('participant')
                          .apply(lambda g: g.sample(frac=subsample_fraction))
                          .index.get_level_values(1))
            clean = clean.loc[sampled_idx].copy()
            # Recompute global_trial index after subsampling
            clean = clean.sort_values(['participant', 'session', 'block', 'trial'])
            clean['global_trial'] = clean.groupby('participant').cumcount()

        # Assign repeats and compute effects
        split = assign_repeats(clean, n_repeats=n_repeats, method=split_method)
        effects = compute_interference_effects(split, task_name)

        # Store both
        results[f'{task_name}_effects'] = effects
        results[f'{task_name}_overall'] = overall

        # Print summary
        n_subj = effects['participant'].nunique()
        repeat_cols = [c for c in effects.columns if c.startswith(f'{task_name}_repeat')]
        print(f"  -> {n_subj} participants × {len(repeat_cols)} repeats")

        # Quick sanity check: print mean effect and trial counts
        mean_effect = effects[repeat_cols].mean().mean() * 1000  # convert to ms
        print(f"  -> Mean interference effect: {mean_effect:.1f} ms")

    # Merge Flanker and Stroop (inner join: keep only participants with both tasks)
    merged = results['flanker_effects'].merge(
        results['stroop_effects'], on='participant', how='inner'
    )
    merged = merged.merge(
        results['flanker_overall'], on='participant', how='inner'
    )
    merged = merged.merge(
        results['stroop_overall'], on='participant', how='inner'
    )

    print(f"\n--- MERGED ---")
    print(f"  {len(merged)} participants with both tasks")
    print(f"  Columns: {merged.columns.tolist()}")

    return merged


# ---------------------------------------------------------------------------
# 6. Diagnostic helpers
# ---------------------------------------------------------------------------

def print_trial_counts(data_root):
    """Print trial count summary for each participant and task (before preprocessing)."""
    data_root = Path(data_root)

    for folder_name in ['Flanker', 'Stroop']:
        folder_path = data_root / folder_name
        if not folder_path.exists():
            print(f"  {folder_name}/ not found")
            continue

        raw = load_task_folder(folder_path)

        counts = raw.groupby(['participant', 'session']).size().unstack(fill_value=0)
        print(f"\n  {folder_name}: trials per participant × session")
        print(f"  {counts.describe().loc[['count','mean','min','max']].to_string()}")

        total = raw.groupby('participant').size()
        print(f"\n  Total trials per participant: "
              f"mean={total.mean():.0f}, min={total.min()}, max={total.max()}")

        # Condition breakdown
        cond_counts = (raw[raw['condition'].isin([0, 2])]
                       .groupby(['participant', 'condition']).size()
                       .unstack(fill_value=0))
        print(f"  Congruent trials per participant: "
              f"mean={cond_counts[0].mean():.0f}, min={cond_counts[0].min()}")
        print(f"  Incongruent trials per participant: "
              f"mean={cond_counts[2].mean():.0f}, min={cond_counts[2].min()}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    # Set your data root here
    data_root = sys.argv[1] if len(sys.argv) > 1 else './hedge_data'

    print("=" * 60)
    print("TRIAL COUNT DIAGNOSTICS")
    print("=" * 60)
    print_trial_counts(data_root)

    print("\n" + "=" * 60)
    print("BUILDING TOOLBOX DATAFRAME (full data, 10 repeats)")
    print("=" * 60)
    df_full = build_toolbox_dataframe(
        data_root, n_repeats=10, split_method='interleaved'
    )

    # Save for the toolbox
    df_full.to_csv('hedge_toolbox_data.csv', index=False)
    print(f"\nSaved to hedge_toolbox_data.csv")
    print(f"\nPreview:")
    print(df_full.head())

    # Also build a 10% subsample version for the ground-truth comparison
    print("\n" + "=" * 60)
    print("BUILDING TOOLBOX DATAFRAME (10% subsample, 2 repeats)")
    print("=" * 60)
    df_sub = build_toolbox_dataframe(
        data_root, n_repeats=2, split_method='interleaved',
        subsample_fraction=0.1, random_state=42
    )

    df_sub.to_csv('hedge_toolbox_data_subsample.csv', index=False)
    print(f"\nSaved to hedge_toolbox_data_subsample.csv")
