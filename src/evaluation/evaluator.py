"""
评估器 (Evaluator)
==================
输入一个已训练模型 + 数据划分,自动跑完三层指标并返回扁平 dict。
扁平 dict 直接进实验注册表,供版本间逐指标对比。

设计要点:
  - 一次评估同时产出 L1/L2/L3 全部指标,保证 "指标多样性"
  - 哪些指标真正参与对比由 config 控制,但默认全算,方便展示
  - 评估逻辑与具体模型解耦,只依赖 BaseRecommender 接口
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data.schema import (
    Dataset,
    ITEM_COL,
    RATING_COL,
    Split,
    USER_COL,
)
from ..models.base import BaseRecommender
from . import metrics as M


class Evaluator:
    def __init__(self, k: int = 10, relevance_threshold: float = 4.0):
        self.k = k
        self.relevance_threshold = relevance_threshold

    def evaluate(
        self, model: BaseRecommender, split: Split, dataset: Dataset
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}

        # ---- L1 评分预测:对测试集逐条预测 ----
        test = split.test
        y_true = test[RATING_COL].to_numpy()
        y_pred = np.array([
            model.predict(u, i)
            for u, i in test[[USER_COL, ITEM_COL]].itertuples(index=False)
        ])
        for name, fn in M.RATING_METRICS.items():
            results[name] = round(fn(y_true, y_pred), 5)

        # ---- 构建每用户的相关物品集合(测试集中高分物品)----
        relevant_by_user: Dict[int, set] = {}
        for u, g in test.groupby(USER_COL):
            rel = set(g[g[RATING_COL] >= self.relevance_threshold][ITEM_COL])
            if rel:
                relevant_by_user[u] = rel

        # ---- L2 Top-N:对有相关物品的用户生成推荐 ----
        # 可靠性护栏:只统计模型真正给出非空推荐的用户,避免"模型无法
        # 推荐"被当成"推荐全错"而拉低指标(两种情况语义不同)。
        all_recs: Dict[int, List[int]] = {}
        rank_acc = {name: [] for name in M.RANKING_METRICS}
        n_eval_users = 0
        n_empty_rec = 0
        rel_sizes: List[int] = []
        for u, relevant in relevant_by_user.items():
            recs = model.recommend(u, k=self.k)
            if not recs:
                n_empty_rec += 1
                continue
            all_recs[u] = recs
            n_eval_users += 1
            rel_sizes.append(len(relevant))
            for name, fn in M.RANKING_METRICS.items():
                rank_acc[name].append(fn(recs, relevant, self.k))
        for name, vals in rank_acc.items():
            key = name.replace("@k", f"@{self.k}")
            results[key] = round(float(np.mean(vals)), 5) if vals else 0.0

        # ---- L3 业务指标 ----
        n_total_items = dataset.n_items
        item_genres = {
            row[ITEM_COL]: set(str(row.get("genres", "")).split("|"))
            for _, row in dataset.items.iterrows()
        }
        item_pop = split.train.groupby(ITEM_COL)[RATING_COL].count().to_dict()
        n_users = split.train[USER_COL].nunique()

        results["coverage"] = round(M.coverage(all_recs, n_total_items), 5)
        results["diversity"] = round(M.diversity(all_recs, item_genres), 5)
        results["novelty"] = round(M.novelty(all_recs, item_pop, n_users), 5)

        # ---- 评估可靠性诊断(随实验记录落盘,供判断 Top-N 指标可信度)----
        # eval_users        : 真正参与 Top-N 评估的用户数
        # eval_empty_rec    : 有相关物品但模型给不出推荐的用户数(召回缺口)
        # eval_rel_size_mean: 平均每用户相关物品数。过小(≈1)会让 precision@k
        #                     被结构性压低、recall 与 hitrate 高度重合,提示
        #                     Top-N 数值在该数据规模下参考价值有限。
        n_relevant_users = len(relevant_by_user)
        results["eval_users"] = n_eval_users
        results["eval_empty_rec"] = n_empty_rec
        results["eval_user_coverage"] = (
            round(n_eval_users / n_relevant_users, 5) if n_relevant_users else 0.0
        )
        results["eval_rel_size_mean"] = (
            round(float(np.mean(rel_sizes)), 3) if rel_sizes else 0.0
        )

        return results

    def evaluate_cold_users(
        self, model: BaseRecommender, split: Split, dataset: Dataset
    ) -> Dict[str, float]:
        """冷启动分层评估:只在 split.cold_user_test(训练集完全未见的用户)上
        计算 Top-N 指标,前缀 cold_。配合常规 evaluate 即"冷启动 vs 老用户"
        分层对比,是误差分析的核心素材。

        关键观察:对冷用户,纯 CF(itemcf/svd)与无画像的 content 会给出空推荐
        (cold_empty_rec 高);popularity / coldstart 能用先验兜底。该差异正是
        冷启动能力的量化体现。
        """
        results: Dict[str, float] = {}
        cold = split.cold_user_test
        if cold is None or len(cold) == 0:
            return results

        relevant_by_user: Dict[int, set] = {}
        for u, g in cold.groupby(USER_COL):
            rel = set(g[g[RATING_COL] >= self.relevance_threshold][ITEM_COL])
            if rel:
                relevant_by_user[u] = rel

        rank_acc = {name: [] for name in M.RANKING_METRICS}
        n_eval = 0
        n_empty = 0
        for u, relevant in relevant_by_user.items():
            recs = model.recommend(u, k=self.k)
            if not recs:
                n_empty += 1
                continue
            n_eval += 1
            for name, fn in M.RANKING_METRICS.items():
                rank_acc[name].append(fn(recs, relevant, self.k))
        for name, vals in rank_acc.items():
            key = "cold_" + name.replace("@k", f"@{self.k}")
            results[key] = round(float(np.mean(vals)), 5) if vals else 0.0

        n_rel_users = len(relevant_by_user)
        results["cold_users"] = n_rel_users
        results["cold_eval_users"] = n_eval
        results["cold_empty_rec"] = n_empty
        results["cold_serve_rate"] = (
            round(n_eval / n_rel_users, 5) if n_rel_users else 0.0
        )
        return results

# Cold-start support marker
