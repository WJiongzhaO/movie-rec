"""
超参网格搜索 (轻量版)
======================
从 YAML 配置读取网格定义 → 笛卡尔积展开 → 逐组合跑实验 → 按目标指标排名。
支持粗筛→细化两阶段搜索。

配置额外字段 (在普通 config 基础上):
  optimize: rmse                  # 排名指标
  fixed_params: {n_epochs: 8}     # 所有组合共享的固定超参 (粗筛省时)
  grid: {n_factors: [20,50,100]}  # 候选超参网格
  refine: {enabled: true, n_between: 1}  # 细化阶段配置
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import yaml

from .runner import run_from_config

_LOWER_BETTER = {"rmse", "mae"}


def expand_grid(grid: Dict[str, Sequence]) -> List[dict]:
    """笛卡尔积展开: {param: [vals]} → [{param: val}, ...]"""
    if not grid:
        return [{}]
    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, c)) for c in combos]


def run_grid(
    config: dict,
    grid: Dict[str, Sequence],
    optimize: str = "rmse",
    fixed_params: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """遍历网格的所有超参组合, 返回按 optimize 排名的结果表。

    每行: {param1, param2, ..., model, metrics...}
    """
    fixed = fixed_params or {}
    model_name = config["model"]["name"]
    combos = expand_grid(grid)

    if verbose:
        print(f"[grid] {model_name}: {len(combos)} 组合, optimize={optimize}")

    rows = []
    for i, params in enumerate(combos, 1):
        full_params = {**fixed, **params}
        cfg = {**config, "model": {"name": model_name, "params": full_params}}
        result = run_from_config(cfg)
        row = {**params}
        row.update(result["metrics"])
        rows.append(row)
        if verbose:
            val = result["metrics"].get(optimize, float("nan"))
            print(f"  [{i}/{len(combos)}] {params} → {optimize}={val:.4f}")

    df = pd.DataFrame(rows)
    ascending = optimize.split("@")[0] in _LOWER_BETTER
    if optimize in df.columns:
        df = df.sort_values(optimize, ascending=ascending).reset_index(drop=True)
    return df


def best_params(df: pd.DataFrame, grid_keys: Sequence[str]) -> dict:
    """从排名后的结果表取最优组合。"""
    if df.empty:
        return {}
    top = df.iloc[0]
    return {k: _to_native(top[k]) for k in grid_keys if k in df.columns}


def refine_grid(
    coarse_grid: Dict[str, Sequence],
    best: dict,
    n_between: int = 1,
) -> Dict[str, Sequence]:
    """围绕粗筛最优点, 在相邻候选之间插值, 生成细化网格。"""
    refined: Dict[str, Sequence] = {}
    for key, candidates in coarse_grid.items():
        cand = [_to_native(c) for c in candidates]
        bv = _to_native(best.get(key))
        if bv is None or not all(isinstance(c, (int, float)) for c in cand):
            refined[key] = [bv] if bv is not None else cand
            continue
        try:
            idx = cand.index(bv)
        except ValueError:
            refined[key] = [bv]
            continue
        lo = cand[idx - 1] if idx > 0 else bv
        hi = cand[idx + 1] if idx < len(cand) - 1 else bv
        points = set()
        for a, b in ((lo, bv), (bv, hi)):
            if a == b:
                points.add(a)
                continue
            for t in range(n_between + 2):
                v = a + (b - a) * t / (n_between + 1)
                points.add(v)
        is_int = all(isinstance(c, int) for c in cand)
        vals = sorted({int(round(p)) if is_int else round(p, 6) for p in points})
        refined[key] = vals
    return refined


def grid_search_from_config(config_path: str | Path, verbose: bool = True) -> dict:
    """从 YAML 配置执行完整的粗筛→细化网格搜索。

    返回:
      {
        "coarse": DataFrame (所有组合 + 指标),
        "coarse_best": dict (最优超参),
        "refine": DataFrame | None,
        "refine_best": dict | None,
      }
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    grid_cfg = cfg.pop("grid", None)
    if not grid_cfg:
        raise ValueError("配置缺少 grid: 区块")

    optimize = cfg.pop("optimize", "rmse")
    fixed = cfg.pop("fixed_params", None) or {}
    refine_cfg = cfg.pop("refine", {}) or {}
    grid_keys = list(grid_cfg)

    # 1) 粗筛
    if verbose:
        print(f"\n===== 粗筛 (coarse) =====")
    coarse_df = run_grid(cfg, grid_cfg, optimize=optimize, fixed_params=fixed,
                         verbose=verbose)
    bp = best_params(coarse_df, grid_keys)
    if verbose:
        print(f"\n[粗筛最优] {bp}")

    # 2) 细化 (可选)
    do_refine = refine_cfg.get("enabled", False)
    refine_df = None
    refine_bp = None
    if do_refine and bp:
        n_between = int(refine_cfg.get("n_between", 1))
        fine_grid = refine_grid(grid_cfg, bp, n_between=n_between)
        if verbose:
            print(f"\n===== 细化 (refine, n_between={n_between}) =====")
            print(f"[细化网格] {fine_grid}")
        refine_df = run_grid(cfg, fine_grid, optimize=optimize, fixed_params=fixed,
                            verbose=verbose)
        refine_bp = best_params(refine_df, list(fine_grid))
        if verbose:
            print(f"\n[细化最优] {refine_bp}")

    return {
        "coarse": coarse_df,
        "coarse_best": bp,
        "refine": refine_df,
        "refine_best": refine_bp,
    }


def _to_native(v):
    if hasattr(v, "item"):
        return v.item()
    return v
