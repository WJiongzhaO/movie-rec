"""
数据清洗模块
============
对 MovieLens 1M 原始数据做质量检测与清洗, 输出清洗报告。
所有操作不修改原始 DataFrame (除非指定 inplace)。

功能:
  - 评分去重 (同用户同物品 → 保留最新)
  - 缺失值统计
  - 低质量用户/物品标记
  - 数据质量综合报告
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .schema import Dataset, ITEM_COL, RATING_COL, TIME_COL, USER_COL


def deduplicate_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """移除同一 (user_id, item_id) 的重复评分, 保留时间戳最新的一条。

    返回: (清洗后的 ratings, 报告 dict)
    """
    n_before = len(ratings)
    dup_mask = ratings.duplicated(subset=[USER_COL, ITEM_COL], keep=False)
    n_dup = dup_mask.sum()

    cleaned = (
        ratings.sort_values(TIME_COL)
        .drop_duplicates(subset=[USER_COL, ITEM_COL], keep="last")
        .reset_index(drop=True)
    )
    n_after = len(cleaned)
    report = {
        "dedup_before": n_before,
        "dedup_after": n_after,
        "dedup_removed": n_before - n_after,
        "dedup_dup_rows": n_dup,
    }
    return cleaned, report


def flag_low_quality(
    ratings: pd.DataFrame,
    min_user_ratings: int = 3,
    min_item_ratings: int = 3,
) -> dict:
    """标记低活跃度用户和冷门物品 (仅标记, 不删除)。

    返回: {low_activity_users: [...], low_popularity_items: [...], ...}
    """
    user_counts = ratings.groupby(USER_COL).size()
    item_counts = ratings.groupby(ITEM_COL).size()

    low_users = user_counts[user_counts < min_user_ratings]
    low_items = item_counts[item_counts < min_item_ratings]

    return {
        "n_total_users": len(user_counts),
        "n_low_activity_users": len(low_users),
        "low_activity_user_ids": low_users.index.tolist(),
        "n_total_items": len(item_counts),
        "n_low_popularity_items": len(low_items),
        "low_popularity_item_ids": low_items.index.tolist(),
        "min_user_ratings_threshold": min_user_ratings,
        "min_item_ratings_threshold": min_item_ratings,
    }


def missing_values_report(dataset: Dataset) -> dict:
    """统计各表的缺失值情况。"""
    report = {}
    for name, df in [("ratings", dataset.ratings), ("items", dataset.items)]:
        nulls = df.isnull().sum()
        report[name] = {
            col: int(nulls[col]) for col in nulls.index if nulls[col] > 0
        }
    if dataset.users is not None:
        nulls = dataset.users.isnull().sum()
        report["users"] = {
            col: int(nulls[col]) for col in nulls.index if nulls[col] > 0
        }
    return report


def rating_distribution_report(ratings: pd.DataFrame) -> dict:
    """评分分布统计。"""
    dist = ratings[RATING_COL].value_counts().sort_index()
    return {
        "mean": float(ratings[RATING_COL].mean()),
        "std": float(ratings[RATING_COL].std()),
        "min": int(ratings[RATING_COL].min()),
        "max": int(ratings[RATING_COL].max()),
        "distribution": {int(k): int(v) for k, v in dist.items()},
    }


def user_rating_stats(ratings: pd.DataFrame) -> dict:
    """每用户评分数统计。"""
    counts = ratings.groupby(USER_COL).size()
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "std": float(counts.std()),
    }


def item_rating_stats(ratings: pd.DataFrame) -> dict:
    """每物品评分数统计。"""
    counts = ratings.groupby(ITEM_COL).size()
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "std": float(counts.std()),
    }


def comprehensive_quality_report(
    dataset: Dataset,
    min_user_ratings: int = 3,
    min_item_ratings: int = 3,
) -> dict:
    """生成综合数据质量报告。

    包含: 去重结果 / 评分分布 / 缺失值 / 低质量标记 / 稀疏度。
    此报告应作为 Dataset.meta 的一部分写入, 供下游模块参考。
    """
    cleaned, dedup_rpt = deduplicate_ratings(dataset.ratings)

    return {
        "dedup": dedup_rpt,
        "rating_distribution": rating_distribution_report(cleaned),
        "missing_values": missing_values_report(dataset),
        "low_quality_flag": flag_low_quality(cleaned, min_user_ratings, min_item_ratings),
        "user_stats": user_rating_stats(cleaned),
        "item_stats": item_rating_stats(cleaned),
        "sparsity": round(dataset.density, 6),
    }
