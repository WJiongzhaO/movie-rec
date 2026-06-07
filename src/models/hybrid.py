"""
模型:混合融合推荐(Hybrid)—— 项目技术亮点
===========================================
加权融合"协同过滤(SVD)"与"基于内容(Content)"两条召回路。

动机:
  - CF 在有行为的热门物品上准,但对冷门/新物品乏力;
  - Content 靠文本特征,能覆盖冷启动,但缺乏群体协同信号。
  两者优势互补,加权融合通常在 Top-N 指标与覆盖率上同时占优。

融合方式(加权式 weighted):
    score = alpha * z(score_cf) + (1 - alpha) * z(score_content)
  其中 z(·) 为对每个用户的候选打分做 min-max 归一化,消除两侧量纲差异。

predict 走评分量纲的凸组合;recommend 走归一化分数融合后排序。
两个子模型均复用已注册实现,接口完全一致。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..data.schema import Dataset, Split
from .base import BaseRecommender
from .content import ContentRecommender
from .svd import SVDRecommender


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(
        self,
        alpha: float = 0.6,
        cf_params: dict | None = None,
        content_params: dict | None = None,
        **params,
    ):
        super().__init__(
            alpha=alpha,
            cf_params=cf_params or {},
            content_params=content_params or {},
            **params,
        )
        self.alpha = alpha
        self.cf = SVDRecommender(**(cf_params or {}))
        self.content = ContentRecommender(**(content_params or {}))

    def fit(self, split: Split, dataset: Dataset) -> "HybridRecommender":
        self.cf.fit(split, dataset)
        self.content.fit(split, dataset)
        self.global_mean_ = self.cf.global_mean_
        # 候选物品全集(并集),推荐时统一打分
        self.all_items_ = np.array(
            sorted(set(self.cf.items_.tolist()) | set(self.content.items_.tolist()))
        )
        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"alpha": self.alpha, "all_items": self.all_items_,
                "cf_state": self.cf._get_state(),
                "ct_state": self.content._get_state()}

    def _set_state(self, state: dict) -> None:
        self.alpha = state["alpha"]
        self.all_items_ = state["all_items"]
        self.cf._set_state(state["cf_state"])
        self.content._set_state(state["ct_state"])
        self.global_mean_ = self.cf.global_mean_
        self._fitted = True

    def predict(self, user_id: int, item_id: int) -> float:
        pc = self.cf.predict(user_id, item_id)
        pk = self.content.predict(user_id, item_id)
        return float(self.alpha * pc + (1 - self.alpha) * pk)

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        items = self.all_items_
        # 两侧分别对全候选向量化打分(避免逐物品 Python 调用)
        cf_scores = self.cf.scores_for_items(user_id, items)
        ct_scores = self.content.scores_for_items(user_id, items)
        fused = self.alpha * _minmax(cf_scores) + (1 - self.alpha) * _minmax(ct_scores)

        # 排除两侧任一记录的已交互物品
        seen = set()
        seen |= self.cf.user_seen_.get(user_id, set())
        seen |= self.content.user_seen_.get(user_id, set())

        order = np.argsort(-fused)
        out = []
        for idx in order:
            it_id = int(items[idx])
            if it_id in seen:
                continue
            out.append(it_id)
            if len(out) >= k:
                break
        return out
