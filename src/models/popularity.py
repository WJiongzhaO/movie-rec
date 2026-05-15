"""
基线模型:Popularity(热门推荐)
================================
最简单的非个性化基线:按物品被评分的次数(或平均分)排序,人人推同一批。
作用是给所有个性化模型一个 "下限参照" —— 如果你的 CF 还不如热门,说明有 bug。
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..data.schema import Dataset, ITEM_COL, RATING_COL, Split, USER_COL
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def fit(self, split: Split, dataset: Dataset) -> "PopularityRecommender":
        train = split.train
        self.global_mean_ = float(train[RATING_COL].mean())
        # 物品平均分 + 评分次数,用 (次数加权的平均分) 作为热门度
        agg = train.groupby(ITEM_COL)[RATING_COL].agg(["mean", "count"])
        # 贝叶斯平滑:避免只被评 1 次的高分物品霸榜
        m, C = self.global_mean_, 5.0
        agg["score"] = (agg["count"] * agg["mean"] + C * m) / (agg["count"] + C)
        self.item_score_ = agg["score"].to_dict()
        self.ranked_items_ = list(
            agg.sort_values("score", ascending=False).index
        )
        # 记录每个用户训练集已交互物品,推荐时排除
        self.user_seen_ = (
            train.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        )
        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"global_mean": self.global_mean_,
                "item_score": self.item_score_,
                "ranked_items": self.ranked_items_,
                "user_seen": self.user_seen_}

    def _set_state(self, state: dict) -> None:
        self.global_mean_ = state["global_mean"]
        self.item_score_ = state["item_score"]
        self.ranked_items_ = state["ranked_items"]
        self.user_seen_ = state["user_seen"]
        self._fitted = True

    def predict(self, user_id: int, item_id: int) -> float:
        return float(self.item_score_.get(item_id, self.global_mean_))

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        seen = self.user_seen_.get(user_id, set())
        out = [i for i in self.ranked_items_ if i not in seen]
        return out[:k]
