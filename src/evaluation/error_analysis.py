"""
错误分析
========
按用户活跃度、物品热度等维度分层分析推荐效果,
回答"谁受益、谁受损"的问题, 是误差分析的核心素材。

功能:
  - by_user_activity(): 按用户评分数量分桶 → 各桶指标
  - by_item_popularity(): 按物品被评分次数分桶 → 各桶命中率
  - rating_error_distribution(): 评分预测误差分布 (过估/低估)
  - per_user_metrics(): 计算每个用户的单独指标值 (用于统计检验)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.schema import Dataset, ITEM_COL, RATING_COL, Split, USER_COL
from ..models.base import BaseRecommender
from . import metrics as M


# ═══════════════════════════════════════════════════════════════════
# 每用户指标 (统计检验基础)
# ═══════════════════════════════════════════════════════════════════

def per_user_rmse(
    model: BaseRecommender, split: Split
) -> Dict[int, float]:
    """计算每个用户在测试集上的 RMSE。"""
    test = split.test
    results: Dict[int, float] = {}
    for u, g in test.groupby(USER_COL):
        y_true = g[RATING_COL].to_numpy()
        y_pred = np.array([model.predict(u, i) for i in g[ITEM_COL]])
        results[int(u)] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return results


def per_user_recall(
    model: BaseRecommender, split: Split, dataset: Dataset,
    k: int = 10, threshold: float = 4.0,
) -> Dict[int, float]:
    """计算每个用户在测试集上的 Recall@K。"""
    test = split.test
    results: Dict[int, float] = {}
    for u, g in test.groupby(USER_COL):
        relevant = set(g[g[RATING_COL] >= threshold][ITEM_COL])
        if not relevant:
            continue
        recs = model.recommend(int(u), k=k)
        if not recs:
            results[int(u)] = 0.0
            continue
        hits = sum(1 for it in recs[:k] if it in relevant)
        results[int(u)] = hits / len(relevant)
    return results


# ═══════════════════════════════════════════════════════════════════
# 分层分析
# ═══════════════════════════════════════════════════════════════════

def by_user_activity(
    model: BaseRecommender, split: Split, dataset: Dataset,
    k: int = 10, threshold: float = 4.0,
    n_buckets: int = 4,
) -> pd.DataFrame:
    """按用户活跃度 (训练集评分数) 分层评估。

    将用户按评分数量均分为 n_buckets 组,
    每组分别计算 RMSE / Recall@K / 推荐覆盖率。
    """
    train = split.train
    # 每用户训练集评分数
    user_activity = train.groupby(USER_COL).size().reset_index(name="n_ratings")
    user_activity["bucket"] = pd.qcut(
        user_activity["n_ratings"], q=n_buckets,
        labels=[f"Q{i+1}" for i in range(n_buckets)],
        duplicates="drop",
    )

    test = split.test
    rows = []
    for bucket_name, grp in user_activity.groupby("bucket"):
        bucket_users = set(grp[USER_COL])
        bucket_test = test[test[USER_COL].isin(bucket_users)]

        # RMSE
        y_true = bucket_test[RATING_COL].to_numpy()
        y_pred = np.array([
            model.predict(int(u), int(i))
            for u, i in bucket_test[[USER_COL, ITEM_COL]].itertuples(index=False)
        ])
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae_val = float(np.mean(np.abs(y_true - y_pred)))

        # Recall@K
        recalls = []
        for u in bucket_users:
            u_test = test[test[USER_COL] == u]
            relevant = set(u_test[u_test[RATING_COL] >= threshold][ITEM_COL])
            if not relevant:
                continue
            recs = model.recommend(int(u), k=k)
            if not recs:
                recalls.append(0.0)
                continue
            hits = sum(1 for it in recs[:k] if it in relevant)
            recalls.append(hits / len(relevant))
        recall_val = float(np.mean(recalls)) if recalls else 0.0

        rows.append({
            "bucket": str(bucket_name),
            "n_users": len(bucket_users),
            "mean_train_ratings": round(float(grp["n_ratings"].mean()), 1),
            "rmse": round(rmse_val, 4),
            "mae": round(mae_val, 4),
            f"recall@{k}": round(recall_val, 4),
            "eval_users": len(recalls),
        })

    return pd.DataFrame(rows)


def by_item_popularity(
    model: BaseRecommender, split: Split, dataset: Dataset,
    k: int = 10, threshold: float = 4.0,
    n_buckets: int = 4,
) -> pd.DataFrame:
    """按物品热度 (训练集被评分次数) 分层评估。

    分析模型在冷门 vs 热门物品上的推荐命中率差异。
    """
    train = split.train
    item_pop = train.groupby(ITEM_COL).size().reset_index(name="n_ratings")
    item_pop["bucket"] = pd.qcut(
        item_pop["n_ratings"], q=n_buckets,
        labels=[f"冷门-Q{i+1}" for i in range(n_buckets)],
        duplicates="drop",
    )

    # 构建用户推荐 + 每个物品的归属桶
    test = split.test
    recommended_by_user: Dict[int, List[int]] = {}
    for u in test[USER_COL].unique():
        recs = model.recommend(int(u), k=k)
        if recs:
            recommended_by_user[int(u)] = recs

    item_bucket_map = dict(zip(item_pop[ITEM_COL], item_pop["bucket"]))

    rows = []
    for bucket_name, grp in item_pop.groupby("bucket"):
        bucket_items = set(grp[ITEM_COL])
        # 只考虑测试集中出现在该桶的物品
        bucket_test = test[test[ITEM_COL].isin(bucket_items)]

        # 推荐命中率: 这些物品被推荐的命中次数
        hits = 0
        total = 0
        for u, g in bucket_test.groupby(USER_COL):
            recs = recommended_by_user.get(int(u), [])
            if not recs:
                continue
            relevant = set(g[g[RATING_COL] >= threshold][ITEM_COL])
            if not relevant:
                continue
            total += 1
            if any(it in relevant for it in recs):
                hits += 1

        rows.append({
            "bucket": str(bucket_name),
            "n_items": len(bucket_items),
            "mean_ratings": round(float(grp["n_ratings"].mean()), 1),
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            "eval_users": total,
        })

    return pd.DataFrame(rows)


def rating_error_distribution(
    model: BaseRecommender, split: Split,
) -> dict:
    """评分预测误差分布分析。

    返回:
      errors: 所有 (y_true - y_pred) 序列
      by_rating: 每个真实评分值的平均误差
      over_est_rate: 过估比例 (pred > true)
      under_est_rate: 低估比例 (pred < true)
      mean_abs_error: MAE
      rmse: RMSE
    """
    test = split.test
    y_true = test[RATING_COL].to_numpy(dtype=np.float32)
    y_pred = np.array([
        model.predict(int(u), int(i))
        for u, i in test[[USER_COL, ITEM_COL]].itertuples(index=False)
    ])

    errors = y_true - y_pred  # >0 = 低估, <0 = 过估
    n = len(errors)

    by_rating = {}
    for r in sorted(test[RATING_COL].unique()):
        mask = test[RATING_COL] == r
        if mask.sum() > 0:
            by_rating[int(r)] = {
                "count": int(mask.sum()),
                "mean_error": round(float(errors[mask].mean()), 4),
                "mae": round(float(np.abs(errors[mask]).mean()), 4),
            }

    return {
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "mean_error": round(float(np.mean(errors)), 4),
        "over_est_rate": round(float(np.mean(y_pred > y_true)), 4),  # 预测高于真实
        "under_est_rate": round(float(np.mean(y_pred < y_true)), 4),  # 预测低于真实
        "exact_rate": round(float(np.mean(np.abs(errors) < 0.5)), 4),
        "by_rating": by_rating,
    }
