"""
模型:冷启动推荐(Cold-Start)
==============================
解决"没有历史行为"的两类冷启动:

  - 新用户冷启动:用户不在训练集中。退路有二——
      (a) 若提供偏好画像(问卷选择的偏好 genres)→ 按内容画像推荐(个性化);
      (b) 否则用 Popularity 兜底(非个性化但稳健)。
  - 新物品冷启动:物品无评分,靠内容特征(genres)被检索,
      天然由内部的 Content 子模型覆盖。

对话式产品里,新用户冷启动常用"偏好问卷 → 首屏推荐",本模型用
`recommend_for_preferences(genres, k)` 直接支持该交互(供 Streamlit 冷启动页调用)。

实现:内部组合 Content(内容画像) + Popularity(兜底),按用户是否已知/
是否有画像自动选择路径。接口与其他模型一致,可直接进实验框架对比。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data.schema import Dataset, ITEM_COL, Split
from .base import BaseRecommender
from .content import ContentRecommender, _tokenize
from .popularity import PopularityRecommender


class ColdStartRecommender(BaseRecommender):
    name = "coldstart"

    def __init__(self, like_threshold: float = 4.0, **params):
        super().__init__(like_threshold=like_threshold, **params)
        self.content = ContentRecommender(like_threshold=like_threshold)
        self.popularity = PopularityRecommender()

    def fit(self, split: Split, dataset: Dataset) -> "ColdStartRecommender":
        self.content.fit(split, dataset)
        self.popularity.fit(split, dataset)
        self.global_mean_ = self.content.global_mean_
        self._known_users_ = set(self.content.user_profiles_.keys())
        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"ct_state": self.content._get_state(),
                "pop_state": self.popularity._get_state(),
                "known_users": self._known_users_}

    def _set_state(self, state: dict) -> None:
        self.content._set_state(state["ct_state"])
        self.popularity._set_state(state["pop_state"])
        self._known_users_ = state["known_users"]
        self.global_mean_ = self.content.global_mean_
        self._fitted = True

    def predict(self, user_id: int, item_id: int) -> float:
        # 已知用户走内容画像,未知用户走流行度先验
        if user_id in self._known_users_:
            return self.content.predict(user_id, item_id)
        return self.popularity.predict(user_id, item_id)

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        # 已知用户:内容个性化;新用户:流行度兜底
        if user_id in self._known_users_:
            recs = self.content.recommend(user_id, k=k)
            if recs:
                return recs
        return self.popularity.recommend(user_id, k=k)

    # ---- 新用户偏好问卷 → 首屏推荐(供 Streamlit 冷启动页直接调用)----
    def recommend_for_preferences(
        self, genres: Sequence[str], k: int = 10,
        exclude: Optional[set] = None,
    ) -> List[int]:
        """根据用户在问卷里选择的偏好 genres,用内容向量构造临时画像并推荐。

        genres 例: ["Action", "Sci-Fi"]。无有效画像时退回热门推荐。
        """
        c = self.content
        if not genres or not getattr(c, "vocab_", None):
            return self.popularity.recommend(-1, k=k)

        # 把偏好 genres 投到 content 的 TF-IDF 词表,构造伪画像向量
        vec = np.zeros(len(c.vocab_), dtype=np.float32)
        for g in genres:
            for tok in _tokenize(g):
                j = c.vocab_.get(tok)
                if j is not None:
                    vec[j] += c.idf_[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return self.popularity.recommend(-1, k=k)
        vec /= norm

        scores = c.item_vecs_ @ vec  # 候选物品与偏好画像的余弦
        exclude = exclude or set()
        order = np.argsort(-scores)
        out = []
        for iidx in order:
            it_id = c.idx2i_[int(iidx)]
            if it_id in exclude:
                continue
            out.append(int(it_id))
            if len(out) >= k:
                break
        return out
