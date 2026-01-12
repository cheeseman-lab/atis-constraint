"""
Utility functions for atis-constraint analysis.
"""

import yaml
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy import stats


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def poisson_ci(observed: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Poisson confidence interval for observed count.

    Uses exact Poisson CI based on chi-square distribution.

    Parameters
    ----------
    observed : int
        Observed count
    confidence : float
        Confidence level (default 0.95)

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of confidence interval
    """
    alpha = 1 - confidence

    if observed == 0:
        lower = 0.0
        upper = stats.chi2.ppf(1 - alpha / 2, 2) / 2
    else:
        lower = stats.chi2.ppf(alpha / 2, 2 * observed) / 2
        upper = stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / 2

    return lower, upper


def oe_poisson_ci(
    observed: int,
    expected: float,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate o/e ratio with Poisson confidence interval.

    Parameters
    ----------
    observed : int
        Observed variant count
    expected : float
        Expected variant count
    confidence : float
        Confidence level (default 0.95)

    Returns
    -------
    Tuple[float, float, float]
        o/e ratio, lower CI, upper CI
    """
    if expected <= 0:
        return np.nan, np.nan, np.nan

    oe = observed / expected
    lower_obs, upper_obs = poisson_ci(observed, confidence)
    lower_oe = lower_obs / expected
    upper_oe = upper_obs / expected

    return oe, lower_oe, upper_oe


def bootstrap_oe_ci(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    n_iterations: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Calculate o/e ratio with bootstrap confidence interval.

    Resamples regions with replacement and recalculates aggregate o/e.

    Parameters
    ----------
    observed_counts : np.ndarray
        Array of observed counts per region
    expected_counts : np.ndarray
        Array of expected counts per region
    n_iterations : int
        Number of bootstrap iterations
    confidence : float
        Confidence level
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[float, float, float]
        o/e ratio, lower CI, upper CI
    """
    rng = np.random.default_rng(random_state)
    n_regions = len(observed_counts)

    total_obs = observed_counts.sum()
    total_exp = expected_counts.sum()

    if total_exp <= 0:
        return np.nan, np.nan, np.nan

    oe = total_obs / total_exp

    bootstrap_oes = []
    for _ in range(n_iterations):
        idx = rng.choice(n_regions, size=n_regions, replace=True)
        boot_obs = observed_counts[idx].sum()
        boot_exp = expected_counts[idx].sum()
        if boot_exp > 0:
            bootstrap_oes.append(boot_obs / boot_exp)

    bootstrap_oes = np.array(bootstrap_oes)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_oes, 100 * alpha / 2)
    upper = np.percentile(bootstrap_oes, 100 * (1 - alpha / 2))

    return oe, lower, upper


def calculate_expected_missense(
    observed_syn: int,
    mis_syn_ratio: float = 2.5
) -> float:
    """
    Calculate expected missense count based on synonymous count.

    Uses genome-wide missense/synonymous ratio as baseline.

    Parameters
    ----------
    observed_syn : int
        Observed synonymous variant count
    mis_syn_ratio : float
        Genome-wide missense to synonymous ratio (default 2.5)

    Returns
    -------
    float
        Expected missense count
    """
    return observed_syn * mis_syn_ratio


def aggregate_by_region_class(
    df,
    class_column: str,
    count_columns: list
) -> dict:
    """
    Aggregate variant counts by region class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variant counts
    class_column : str
        Column name for region class
    count_columns : list
        List of column names to aggregate

    Returns
    -------
    dict
        Dictionary with aggregated counts per class
    """
    result = {}
    for region_class in df[class_column].unique():
        mask = df[class_column] == region_class
        class_data = {
            "n_regions": mask.sum(),
        }
        for col in count_columns:
            class_data[col] = df.loc[mask, col].sum()
        result[region_class] = class_data

    return result


def permutation_test(
    group1_obs: np.ndarray,
    group1_exp: np.ndarray,
    group2_obs: np.ndarray,
    group2_exp: np.ndarray,
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation test for difference in o/e between two groups.

    Parameters
    ----------
    group1_obs, group1_exp : np.ndarray
        Observed and expected counts for group 1
    group2_obs, group2_exp : np.ndarray
        Observed and expected counts for group 2
    n_permutations : int
        Number of permutations
    random_state : int, optional
        Random seed

    Returns
    -------
    Tuple[float, float]
        Observed difference in o/e, p-value
    """
    rng = np.random.default_rng(random_state)

    oe1 = group1_obs.sum() / group1_exp.sum()
    oe2 = group2_obs.sum() / group2_exp.sum()
    observed_diff = oe1 - oe2

    all_obs = np.concatenate([group1_obs, group2_obs])
    all_exp = np.concatenate([group1_exp, group2_exp])
    n1 = len(group1_obs)

    null_diffs = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(all_obs))
        perm_obs1 = all_obs[idx[:n1]]
        perm_exp1 = all_exp[idx[:n1]]
        perm_obs2 = all_obs[idx[n1:]]
        perm_exp2 = all_exp[idx[n1:]]

        perm_oe1 = perm_obs1.sum() / perm_exp1.sum()
        perm_oe2 = perm_obs2.sum() / perm_exp2.sum()
        null_diffs.append(perm_oe1 - perm_oe2)

    null_diffs = np.array(null_diffs)
    p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).mean()

    return observed_diff, p_value
