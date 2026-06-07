"""
模型对比
========
输入两个实验结果 dict, 逐指标输出 delta / 百分比 / 优劣判定。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

# 越小越好的指标
_LOWER_BETTER = {"rmse", "mae"}

# 诊断字段: 不参与优劣判定
_DIAGNOSTIC = {
    "eval_users", "eval_empty_rec", "eval_user_coverage",
    "eval_rel_size_mean", "cold_users", "cold_eval_users", "cold_empty_rec",
}


def _direction(metric: str) -> int:
    base = metric.split("@")[0]
    return -1 if base in _LOWER_BETTER else 1


def compare(run_a: dict, run_b: dict) -> pd.DataFrame:
    """对比两次实验的逐指标差异。

    run_a 为基线, run_b 为对比对象。
    返回 DataFrame: metric | baseline | target | delta | delta_pct | judgment
    """
    m_a = run_a["metrics"]
    m_b = run_b["metrics"]
    all_keys = sorted(set(m_a) | set(m_b))

    rows = []
    for key in all_keys:
        va = m_a.get(key)
        vb = m_b.get(key)
        row: dict = {"metric": key, "baseline": va, "target": vb}
        if va is not None and vb is not None and va != 0:
            delta = vb - va
            pct = round(delta / abs(va) * 100, 2)
            row["delta"] = round(delta, 5)
            row["delta_pct"] = pct
            if key in _DIAGNOSTIC:
                row["judgment"] = "(诊断)"
            else:
                better = _direction(key) * delta > 0
                row["judgment"] = "↑ better" if better else ("→ tie" if delta == 0 else "↓ worse")
        else:
            row["delta"] = None
            row["delta_pct"] = None
            row["judgment"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


def compare_many(runs: List[dict], baseline_idx: int = 0) -> pd.DataFrame:
    """多实验对比, 以第一个为基线。"""
    if len(runs) < 2:
        raise ValueError("至少需要 2 个实验才能对比")
    base = runs[baseline_idx]
    frames = []
    for i, r in enumerate(runs):
        if i == baseline_idx:
            continue
        label = r.get("model", f"run_{i}")
        df = compare(base, r)
        df = df.rename(columns={
            "target": label, "delta": f"{label}_Δ",
            "delta_pct": f"{label}_Δ%", "judgment": f"{label}_判定",
        })
        frames.append(df)
    result = frames[0]
    for df in frames[1:]:
        result = result.merge(
            df[["metric", *[c for c in df.columns if c != "metric"]]],
            on="metric", how="outer")
    return result
