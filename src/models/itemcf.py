"""
基线模型:ItemCF(基于物品的协同过滤)
=====================================
经典个性化基线。思路:
  1. 用评分构建 user-item 矩阵
  2. 计算物品两两余弦相似度
  3. 预测 = 用户已评物品中,与目标物品相似者的加权平均分
  4. 推荐 = 对未交互物品按预测分排序取 Top-K

实现刻意保持轻量(纯 numpy/pandas),mock 规模下秒级跑完。相似度矩阵在 fit 阶段
做 top-N 邻居截断(每个物品只保留 topk_neighbors 个最相似邻居),既降噪又提速;
真实 1M 规模时可在此基础上换成稀疏矩阵进一步优化,接口不变。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..data.schema import (
    Dataset,
    ITEM_COL,
    RATING_COL,
    Split,
    USER_COL,
)
from .base import BaseRecommender


class ItemCFRecommender(BaseRecommender):
    name = "itemcf"

    def __init__(self, topk_neighbors: int = 50, **params):
        super().__init__(topk_neighbors=topk_neighbors, **params)
        self.topk_neighbors = topk_neighbors

    def fit(self, split: Split, dataset: Dataset) -> "ItemCFRecommender":
        train = split.train
        self.global_mean_ = float(train[RATING_COL].mean())

        # 建立 id <-> 索引映射
        self.users_ = train[USER_COL].unique()
        self.items_ = train[ITEM_COL].unique()
        self.u2idx_ = {u: i for i, u in enumerate(self.users_)}
        self.i2idx_ = {it: i for i, it in enumerate(self.items_)}
        self.idx2i_ = {i: it for it, i in self.i2idx_.items()}

        n_u, n_i = len(self.users_), len(self.items_)
        mat = np.zeros((n_u, n_i), dtype=np.float32)
        for u, it, r in train[[USER_COL, ITEM_COL, RATING_COL]].itertuples(index=False):
            mat[self.u2idx_[u], self.i2idx_[it]] = r
        self.matrix_ = mat

        # 物品余弦相似度(列向量两两余弦)
        norms = np.linalg.norm(mat, axis=0) + 1e-9
        normalized = mat / norms
        self.sim_ = normalized.T @ normalized  # (n_i, n_i)
        np.fill_diagonal(self.sim_, 0.0)

        # top-N neighbors 截断:每个物品只保留相似度最高的 topk_neighbors 个邻居,
        # 其余置 0。这才是 ItemCF 的核心超参——既降噪(剔除弱相关物品)又提速。
        # 此前版本计算了 topk_neighbors 却从不使用,导致该超参对结果零影响。
        k_nb = int(self.topk_neighbors)
        if 0 < k_nb < n_i - 1:
            # 对每行(每个物品)保留 top-k 列,其余清零
            keep = np.argpartition(-self.sim_, k_nb, axis=1)[:, :k_nb]
            mask = np.zeros_like(self.sim_, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            self.sim_ = np.where(mask, self.sim_, 0.0).astype(np.float32)

        # 用户已评物品(索引)与原始评分,供预测/排除使用
        self.user_rated_ = {
            self.u2idx_[u]: g.set_index(ITEM_COL)[RATING_COL].to_dict()
            for u, g in train.groupby(USER_COL)
        }
        self.user_seen_ = (
            train.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        )
        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"global_mean": self.global_mean_, "users": self.users_,
                "items": self.items_, "u2idx": self.u2idx_, "i2idx": self.i2idx_,
                "idx2i": self.idx2i_, "matrix": self.matrix_, "sim": self.sim_,
                "user_rated": self.user_rated_, "user_seen": self.user_seen_}

    def _set_state(self, state: dict) -> None:
        self.global_mean_ = state["global_mean"]; self.users_ = state["users"]
        self.items_ = state["items"]; self.u2idx_ = state["u2idx"]
        self.i2idx_ = state["i2idx"]; self.idx2i_ = state["idx2i"]
        self.matrix_ = state["matrix"]; self.sim_ = state["sim"]
        self.user_rated_ = state["user_rated"]; self.user_seen_ = state["user_seen"]
        self._fitted = True

    def _predict_idx(self, uidx: int, iidx: int) -> float:
        rated = self.user_rated_.get(uidx, {})
        if not rated:
            return self.global_mean_
        sims, scores = [], []
        for it_id, r in rated.items():
            j = self.i2idx_.get(it_id)
            if j is None:
                continue
            s = self.sim_[iidx, j]
            if s > 0:
                sims.append(s)
                scores.append(s * r)
        if not sims:
            return self.global_mean_
        return float(np.sum(scores) / (np.sum(sims) + 1e-9))

    def predict(self, user_id: int, item_id: int) -> float:
        uidx = self.u2idx_.get(user_id)
        iidx = self.i2idx_.get(item_id)
        if uidx is None or iidx is None:
            return self.global_mean_
        return self._predict_idx(uidx, iidx)

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        uidx = self.u2idx_.get(user_id)
        if uidx is None:
            return []
        rated = self.user_rated_.get(uidx, {})
        if not rated:
            return []
        # 用户向量 · 相似度矩阵 = 对所有物品的打分(高效批量)
        user_vec = self.matrix_[uidx]
        scores = self.sim_ @ user_vec  # (n_i,)
        seen = self.user_seen_.get(user_id, set())
        order = np.argsort(-scores)
        out = []
        for iidx in order:
            it_id = self.idx2i_[iidx]
            if it_id in seen:
                continue
            out.append(int(it_id))
            if len(out) >= k:
                break
        return out
