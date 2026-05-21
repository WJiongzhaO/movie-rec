"""
模型:SVD 矩阵分解(FunkSVD / 偏置 SGD)
=======================================
针对评分矩阵稀疏(MovieLens 1M 密度 ~4.2%)的主力协同过滤算法。

模型形式(带偏置的隐因子模型):
    r_hat(u, i) = mu + b_u + b_i + p_u · q_i

其中:
  - mu      : 全局平均分
  - b_u/b_i : 用户/物品偏置
  - p_u/q_i : k 维隐因子向量

用带 L2 正则的小批量随机梯度下降(mini-batch SGD,纯 numpy 向量化)在
训练集观测评分上拟合;批内用 np.add.at 做 scatter-add 累加梯度,既保持
SGD 语义又避免逐样本 Python 循环,真实 1M(80 万条)秒级完成单轮。
相比 ItemCF:
  - 不需存储 O(n_i^2) 的相似度矩阵,内存只需 O((n_u+n_i)·k)
  - 隐因子能捕捉潜在语义,稀疏场景下泛化更好
接口与其他模型完全一致,直接纳入实验框架对比。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..data.schema import (
    Dataset,
    ITEM_COL,
    RATING_COL,
    Split,
    USER_COL,
)
from .base import BaseRecommender


class SVDRecommender(BaseRecommender):
    name = "svd"

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
        **params,
    ):
        super().__init__(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr=lr,
            reg=reg,
            seed=seed,
            **params,
        )
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, split: Split, dataset: Dataset) -> "SVDRecommender":
        train = split.train
        rng = np.random.default_rng(self.seed)

        self.global_mean_ = float(train[RATING_COL].mean())

        self.users_ = train[USER_COL].unique()
        self.items_ = train[ITEM_COL].unique()
        self.u2idx_ = {u: i for i, u in enumerate(self.users_)}
        self.i2idx_ = {it: i for i, it in enumerate(self.items_)}
        self.idx2i_ = {i: it for it, i in self.i2idx_.items()}

        n_u, n_i = len(self.users_), len(self.items_)
        k = self.n_factors

        # 隐因子小幅随机初始化;偏置置零
        self.P_ = rng.normal(0, 0.1, (n_u, k)).astype(np.float32)
        self.Q_ = rng.normal(0, 0.1, (n_i, k)).astype(np.float32)
        self.bu_ = np.zeros(n_u, dtype=np.float32)
        self.bi_ = np.zeros(n_i, dtype=np.float32)

        # 预先转成索引数组,SGD 内循环纯 numpy 标量运算
        u_idx = train[USER_COL].map(self.u2idx_).to_numpy()
        i_idx = train[ITEM_COL].map(self.i2idx_).to_numpy()
        ratings = train[RATING_COL].to_numpy(dtype=np.float32)
        n = len(ratings)

        mu = self.global_mean_
        lr, reg = self.lr, self.reg
        batch = int(self.params.get("batch_size", 2048))
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                u = u_idx[idx]
                it = i_idx[idx]
                r = ratings[idx]
                pred = (
                    mu
                    + self.bu_[u]
                    + self.bi_[it]
                    + np.einsum("ij,ij->i", self.P_[u], self.Q_[it])
                )
                err = (r - pred).astype(np.float32)
                # 偏置更新(同一 u/it 在批内可能重复,用 np.add.at 累加)
                np.add.at(self.bu_, u, lr * (err - reg * self.bu_[u]))
                np.add.at(self.bi_, it, lr * (err - reg * self.bi_[it]))
                # 隐因子更新:先算梯度,再 scatter-add(避免批内别名问题)
                pu = self.P_[u]
                qi = self.Q_[it]
                grad_p = err[:, None] * qi - reg * pu
                grad_q = err[:, None] * pu - reg * qi
                np.add.at(self.P_, u, lr * grad_p)
                np.add.at(self.Q_, it, lr * grad_q)

        # 已交互物品(用于推荐时排除)
        self.user_seen_ = (
            train.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        )
        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"global_mean": self.global_mean_, "users": self.users_,
                "items": self.items_, "u2idx": self.u2idx_, "i2idx": self.i2idx_,
                "idx2i": self.idx2i_, "P": self.P_, "Q": self.Q_,
                "bu": self.bu_, "bi": self.bi_, "user_seen": self.user_seen_}

    def _set_state(self, state: dict) -> None:
        self.global_mean_ = state["global_mean"]; self.users_ = state["users"]
        self.items_ = state["items"]; self.u2idx_ = state["u2idx"]
        self.i2idx_ = state["i2idx"]; self.idx2i_ = state["idx2i"]
        self.P_ = state["P"]; self.Q_ = state["Q"]
        self.bu_ = state["bu"]; self.bi_ = state["bi"]
        self.user_seen_ = state["user_seen"]; self._fitted = True

    def predict(self, user_id: int, item_id: int) -> float:
        uidx = self.u2idx_.get(user_id)
        iidx = self.i2idx_.get(item_id)
        mu = self.global_mean_
        if uidx is None and iidx is None:
            return mu
        if uidx is None:
            return float(np.clip(mu + self.bi_[iidx], 1.0, 5.0))
        if iidx is None:
            return float(np.clip(mu + self.bu_[uidx], 1.0, 5.0))
        pred = mu + self.bu_[uidx] + self.bi_[iidx] + self.P_[uidx] @ self.Q_[iidx]
        return float(np.clip(pred, 1.0, 5.0))

    def scores_for_items(self, user_id: int, item_ids) -> np.ndarray:
        """向量化:返回该用户对给定 item_ids 的预测分数(评分量纲)。
        未知 user/item 回退到对应偏置/全局均值。供 Hybrid 批量融合使用。
        """
        mu = self.global_mean_
        uidx = self.u2idx_.get(user_id)
        out = np.full(len(item_ids), mu, dtype=np.float32)
        iidxs = np.array([self.i2idx_.get(int(it), -1) for it in item_ids])
        known = iidxs >= 0
        if uidx is None:
            out[known] = mu + self.bi_[iidxs[known]]
            return np.clip(out, 1.0, 5.0)
        out += self.bu_[uidx]
        if known.any():
            ki = iidxs[known]
            out[known] = (
                mu + self.bu_[uidx] + self.bi_[ki] + self.Q_[ki] @ self.P_[uidx]
            )
        return np.clip(out, 1.0, 5.0)

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        uidx = self.u2idx_.get(user_id)
        if uidx is None:
            return []
        # 对所有物品批量打分: mu + b_u + b_i + p_u·Q
        scores = self.global_mean_ + self.bu_[uidx] + self.bi_ + self.Q_ @ self.P_[uidx]
        seen = self.user_seen_.get(user_id, set())
        order = np.argsort(-scores)
        out = []
        for iidx in order:
            it_id = self.idx2i_[int(iidx)]
            if it_id in seen:
                continue
            out.append(int(it_id))
            if len(out) >= k:
                break
        return out
