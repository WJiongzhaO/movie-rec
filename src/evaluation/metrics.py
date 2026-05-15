"""
指标库 (可插拔、多样化)
========================
对齐提案的 "三层评估" 框架,把指标按层组织,全部走统一接口,后续加新指标
只需写一个函数并在 METRICS 里登记一行 —— 这就是 "指标多样性 + 易迭代" 的落地。

三层:
  L1 评分预测 : RMSE, MAE
  L2 Top-N 排序 : Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K
  L3 业务指标 : Coverage, Diversity, Novelty

每个指标是一个纯函数,签名见下方注释。Evaluator 负责调度与汇总。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

import numpy as np

# ============ L1: 评分预测指标 ============
# 签名: f(y_true: array, y_pred: array) -> float


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


# ============ L2: Top-N 排序指标 ============
# 签名: f(recommended: List[int], relevant: set, k: int) -> float


def precision_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for it in rec_k if it in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for it in rec_k if it in relevant)
    return hits / len(relevant)


def hitrate_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    return 1.0 if any(it in relevant for it in rec_k) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(idx + 2)
        for idx, it in enumerate(rec_k)
        if it in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits, score = 0, 0.0
    for idx, it in enumerate(rec_k):
        if it in relevant:
            hits += 1
            score += hits / (idx + 1)
    return score / min(len(relevant), k)


# ============ L3: 业务指标(对全体推荐列表聚合)============
# 签名: f(all_recs: Dict[user, List[item]], context) -> float


def coverage(all_recs: Dict[int, List[int]], n_total_items: int) -> float:
    """覆盖率:被推荐到的不同物品数 / 物品总数。"""
    if n_total_items == 0:
        return 0.0
    recommended_items = set()
    for items in all_recs.values():
        recommended_items.update(items)
    return len(recommended_items) / n_total_items


def diversity(all_recs: Dict[int, List[int]], item_genres: Dict[int, set]) -> float:
    """多样性:推荐列表内物品两两类别 Jaccard 距离的平均。越高越多样。"""
    dists = []
    for items in all_recs.values():
        if len(items) < 2:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ga = item_genres.get(items[a], set())
                gb = item_genres.get(items[b], set())
                union = ga | gb
                jacc = len(ga & gb) / len(union) if union else 0.0
                dists.append(1 - jacc)
    return float(np.mean(dists)) if dists else 0.0


def novelty(all_recs: Dict[int, List[int]], item_popularity: Dict[int, int],
            n_users: int) -> float:
    """新颖度:推荐物品的平均 self-information(-log2 流行度)。越高越冷门。"""
    if n_users == 0:
        return 0.0
    vals = []
    for items in all_recs.values():
        for it in items:
            pop = item_popularity.get(it, 1)
            p = pop / n_users
            vals.append(-math.log2(p)) if p > 0 else None
    return float(np.mean(vals)) if vals else 0.0


# ============ 指标注册中心 ============
RATING_METRICS: Dict[str, Callable] = {"rmse": rmse, "mae": mae}
RANKING_METRICS: Dict[str, Callable] = {
    "precision@k": precision_at_k,
    "recall@k": recall_at_k,
    "hitrate@k": hitrate_at_k,
    "ndcg@k": ndcg_at_k,
    "map@k": map_at_k,
}
BUSINESS_METRICS: Dict[str, Callable] = {
    "coverage": coverage,
    "diversity": diversity,
    "novelty": novelty,
}
