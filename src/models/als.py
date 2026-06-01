"""
ALS 交替最小二乘 (Alternating Least Squares)
==============================================
与 SVD 的 SGD 优化不同, ALS 用闭式解交替更新用户/物品隐因子,
天然支持并行化, 在隐式反馈场景下尤为经典。

算法:
  min ∑ (r_ui - p_u·q_i)² + λ(||p_u||² + ||q_i||²)

每轮:
  1. 固定 Q, 对每个用户 u 解 ridge 回归 → p_u
  2. 固定 P, 对每个物品 i 解 ridge 回归 → q_i

收敛快、无需调学习率, 是 SVD 的重要对比基线。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..data.schema import Dataset, ITEM_COL, RATING_COL, Split, USER_COL
from .base import BaseRecommender


class ALSRecommender(BaseRecommender):
    name = "als"

    def __init__(self, n_factors: int = 50, reg: float = 0.1,
                 n_epochs: int = 15, seed: int = 42, **params):
        super().__init__(n_factors=n_factors, reg=reg, n_epochs=n_epochs,
                         seed=seed, **params)
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

    # ── 状态导出 (供持久化) ──
    def _get_state(self) -> dict:
        return {
            "P": self.P_, "Q": self.Q_, "bu": self.bu_, "bi": self.bi_,
            "global_mean": self.global_mean_,
            "u2idx": self.u2idx_, "i2idx": self.i2idx_, "idx2i": self.idx2i_,
            "user_seen": self.user_seen_,
        }

    def _set_state(self, state: dict) -> None:
        self.P_ = state["P"]; self.Q_ = state["Q"]
        self.bu_ = state["bu"]; self.bi_ = state["bi"]
        self.global_mean_ = state["global_mean"]
        self.u2idx_ = state["u2idx"]; self.i2idx_ = state["i2idx"]
        self.idx2i_ = state["idx2i"]; self.user_seen_ = state["user_seen"]
        self._fitted = True

    # ── 训练 ──
    def fit(self, split: Split, dataset: Dataset) -> "ALSRecommender":
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
        lam = self.reg

        # 隐因子小幅随机初始化; 偏置从全局均值起步
        self.P_ = rng.normal(0, 0.1, (n_u, k)).astype(np.float32)
        self.Q_ = rng.normal(0, 0.1, (n_i, k)).astype(np.float32)
        self.bu_ = np.zeros(n_u, dtype=np.float32)
        self.bi_ = np.zeros(n_i, dtype=np.float32)

        # 构建每用户的 (item_idx, rating) 列表和每物品的 (user_idx, rating) 列表
        u_items: list[tuple[np.ndarray, np.ndarray]] = [None] * n_u  # type: ignore
        i_users: list[tuple[np.ndarray, np.ndarray]] = [None] * n_i  # type: ignore

        for _, row in train.iterrows():
            u = int(row[USER_COL]); it = int(row[ITEM_COL]); r = float(row[RATING_COL])
            ui = self.u2idx_[u]; ii = self.i2idx_[it]
            if u_items[ui] is None:
                u_items[ui] = ([], [])
            u_items[ui][0].append(ii); u_items[ui][1].append(r)  # type: ignore
            if i_users[ii] is None:
                i_users[ii] = ([], [])
            i_users[ii][0].append(ui); i_users[ii][1].append(r)  # type: ignore

        # 转 numpy
        for ui in range(n_u):
            if u_items[ui] is not None:
                u_items[ui] = (np.array(u_items[ui][0], dtype=np.int32),
                              np.array(u_items[ui][1], dtype=np.float32))
        for ii in range(n_i):
            if i_users[ii] is not None:
                i_users[ii] = (np.array(i_users[ii][0], dtype=np.int32),
                              np.array(i_users[ii][1], dtype=np.float32))

        # ALS 迭代
        mu = self.global_mean_
        eye_k = np.eye(k, dtype=np.float32)

        for epoch in range(self.n_epochs):
            # ── 固定 Q, 更新 P 和 bu ──
            for ui in range(n_u):
                if u_items[ui] is None:
                    continue
                ii_idx, ratings = u_items[ui]
                Qi = self.Q_[ii_idx]          # (m, k)
                bi = self.bi_[ii_idx]         # (m,)
                target = ratings - mu - bi     # (m,)

                A = Qi.T @ Qi + lam * eye_k    # (k, k)
                b = Qi.T @ target              # (k,)
                self.P_[ui] = np.linalg.solve(A, b).astype(np.float32)

                # 偏置更新: b_u = mean(r - mu - b_i - p_u·q_i)
                pred_pq = self.P_[ui] @ Qi.T
                self.bu_[ui] = float(np.mean(target - pred_pq))

            # ── 固定 P, 更新 Q 和 bi ──
            for ii in range(n_i):
                if i_users[ii] is None:
                    continue
                ui_idx, ratings = i_users[ii]
                Pu = self.P_[ui_idx]           # (m, k)
                bu_val = self.bu_[ui_idx]      # (m,)
                target = ratings - mu - bu_val  # (m,)

                A = Pu.T @ Pu + lam * eye_k     # (k, k)
                b = Pu.T @ target               # (k,)
                self.Q_[ii] = np.linalg.solve(A, b).astype(np.float32)

                pred_pq = self.Q_[ii] @ Pu.T
                self.bi_[ii] = float(np.mean(target - pred_pq))

        self.user_seen_ = (
            train.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        )
        self._fitted = True
        return self

    # ── 预测 / 推荐 ──
    def predict(self, user_id: int, item_id: int) -> float:
        mu = self.global_mean_
        ui = self.u2idx_.get(user_id); ii = self.i2idx_.get(item_id)
        if ui is None and ii is None:
            return mu
        if ui is None:
            return float(np.clip(mu + self.bi_[ii], 1.0, 5.0))
        if ii is None:
            return float(np.clip(mu + self.bu_[ui], 1.0, 5.0))
        pred = mu + self.bu_[ui] + self.bi_[ii] + self.P_[ui] @ self.Q_[ii]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        ui = self.u2idx_.get(user_id)
        if ui is None:
            return []
        scores = (self.global_mean_ + self.bu_[ui] + self.bi_
                  + self.Q_ @ self.P_[ui])
        seen = self.user_seen_.get(user_id, set())
        order = np.argsort(-scores)
        out = []
        for idx in order:
            it_id = self.idx2i_[int(idx)]
            if it_id in seen:
                continue
            out.append(int(it_id))
            if len(out) >= k:
                break
        return out

    def scores_for_items(self, user_id: int, item_ids) -> np.ndarray:
        """向量化打分 (供 Hybrid 融合使用)。"""
        mu = self.global_mean_
        ui = self.u2idx_.get(user_id)
        out = np.full(len(item_ids), mu, dtype=np.float32)
        iidxs = np.array([self.i2idx_.get(int(it), -1) for it in item_ids])
        known = iidxs >= 0
        if ui is None:
            out[known] = mu + self.bi_[iidxs[known]]
            return np.clip(out, 1.0, 5.0)
        out += self.bu_[ui]
        if known.any():
            ki = iidxs[known]
            out[known] = mu + self.bu_[ui] + self.bi_[ki] + self.Q_[ki] @ self.P_[ui]
        return np.clip(out, 1.0, 5.0)
