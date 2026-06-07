"""
统计显著性检验
==============
为模型对比提供定量显著性证据, 支撑"提升是否显著"的判断。

功能:
  - paired_ttest(): 配对 t 检验 (两模型在同一用户上的指标差异)
  - bootstrap_ci(): Bootstrap 置信区间
  - cohens_d(): Cohen's d 效应量
  - compare_models(): 一站式对比 (t-test + bootstrap + effect size)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.random import default_rng


def paired_ttest(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """配对 t 检验: H0 = 两模型均值无差异。

    参数:
      vals_a, vals_b: 每个样本在两个模型上的指标值 (长度相同)
      alpha: 显著性水平

    返回: {t_statistic, p_value, significant, mean_diff, ci_95}
    """
    n = len(vals_a)
    if n < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False,
                "mean_diff": 0.0, "ci_95": (0.0, 0.0)}

    diffs = vals_a - vals_b
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se_d = std_d / np.sqrt(n)
    t_stat = mean_d / se_d if se_d > 1e-9 else 0.0

    # 双侧 p 值 (Student's t 分布近似, n>30 时接近正态)
    from math import erf, sqrt
    def _t_cdf(t, df):
        """Student's t CDF 近似 (适用于 df>1)。"""
        x = t * (1 - 1/(4*df)) / sqrt(1 + t*t/(2*df))
        return 0.5 * (1 + erf(x / sqrt(2))) if df > 1 else 0.5

    df = n - 1
    p_one = 1 - _t_cdf(abs(t_stat), df)
    p_val = 2 * p_one

    # 95% CI
    t_crit = 1.96 if n > 30 else _t_inv(0.975, df)
    ci_lo = mean_d - t_crit * se_d
    ci_hi = mean_d + t_crit * se_d

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < alpha,
        "alpha": alpha,
        "mean_diff": round(mean_d, 6),
        "ci_95": (round(ci_lo, 6), round(ci_hi, 6)),
        "n_pairs": n,
    }


def _t_inv(p: float, df: int, tol: float = 1e-6) -> float:
    """Student's t 分位数 (二分搜索)。"""
    from math import erf, sqrt
    def _cdf(t):
        x = t * (1 - 1/(4*df)) / sqrt(1 + t*t/(2*df))
        return 0.5 * (1 + erf(x / sqrt(2)))
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def bootstrap_ci(
    vals: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 置信区间 (适用于任意指标)。

    返回: {mean, std, ci_lower, ci_upper, n_bootstrap}
    """
    rng = default_rng(seed)
    n = len(vals)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(vals[idx])

    alpha = 1 - ci_level
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))

    return {
        "mean": round(float(np.mean(vals)), 6),
        "std": round(float(np.std(vals, ddof=1)), 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


def cohens_d(vals_a: np.ndarray, vals_b: np.ndarray) -> dict:
    """Cohen's d 效应量: |d|<0.2 可忽略; 0.2~0.5 小; 0.5~0.8 中; >0.8 大。

    使用 pooled SD。
    """
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a < 2 or n_b < 2:
        return {"cohens_d": 0.0, "magnitude": "—", "mean_diff": 0.0}

    m_a, m_b = np.mean(vals_a), np.mean(vals_b)
    s_a = np.std(vals_a, ddof=1)
    s_b = np.std(vals_b, ddof=1)

    # pooled SD
    sp = np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    d = abs(m_a - m_b) / sp if sp > 1e-9 else 0.0

    if d < 0.2:     mag = "negligible"
    elif d < 0.5:   mag = "small"
    elif d < 0.8:   mag = "medium"
    else:           mag = "large"

    return {
        "cohens_d": round(d, 4),
        "magnitude": mag,
        "mean_diff": round(float(m_a - m_b), 6),
    }


def compare_models_per_user(
    model_a_vals: Dict[int, float],
    model_b_vals: Dict[int, float],
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> dict:
    """一站式统计检验: 以用户为配对单位, 比较两模型在某指标上的差异。

    参数:
      model_a_vals: {user_id: metric_value} (如每用户 RMSE 或 recall)
      model_b_vals: {user_id: metric_value}

    返回: {ttest, bootstrap, cohens_d}
    """
    common_users = sorted(set(model_a_vals) & set(model_b_vals))
    if len(common_users) < 2:
        return {"error": "Not enough common users for comparison"}

    va = np.array([model_a_vals[u] for u in common_users])
    vb = np.array([model_b_vals[u] for u in common_users])

    return {
        "ttest": paired_ttest(va, vb, alpha=alpha),
        "bootstrap_a": bootstrap_ci(va, n_bootstrap=n_bootstrap),
        "bootstrap_b": bootstrap_ci(vb, n_bootstrap=n_bootstrap),
        "cohens_d": cohens_d(va, vb),
    }
