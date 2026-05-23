"""
模型:基于内容的推荐(Content-Based)
=====================================
不依赖协同信号,而是用物品自身的文本特征(genres)构造物品画像,
再用用户历史高分物品聚合出"用户画像",按余弦相似度推荐。

特征构造:
  - 文本 = genres(经 IMDb 融合的并集,信息量显著强于单源)
  - TF-IDF 向量化(纯 numpy 实现,无 sklearn 依赖,保持轻量)

价值:
  - 缓解冷启动(新物品只要有文本即可被推荐)
  - 与 CF 互补,是 Hybrid 融合的内容侧来源
接口与其他模型一致。
"""
from __future__ import annotations

import re
from collections import Counter
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

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # genres 用 '|' 分隔;统一小写并按字母数字切词
    return _TOKEN_RE.findall(str(text).lower())


class ContentRecommender(BaseRecommender):
    name = "content"

    def __init__(self, like_threshold: float = 4.0, **params):
        super().__init__(like_threshold=like_threshold, **params)
        self.like_threshold = like_threshold

    def _build_item_text(self, dataset: Dataset) -> dict:
        items = dataset.items
        texts = {}
        for row in items.itertuples(index=False):
            rd = row._asdict()
            genres = str(rd.get("genres", ""))
            texts[rd[ITEM_COL]] = genres
        return texts

    def fit(self, split: Split, dataset: Dataset) -> "ContentRecommender":
        train = split.train
        self.global_mean_ = float(train[RATING_COL].mean())

        item_text = self._build_item_text(dataset)
        # 只为训练集中出现过的物品建画像(评估时未出现的物品无意义)
        self.items_ = np.array(list(item_text.keys()))
        self.i2idx_ = {it: i for i, it in enumerate(self.items_)}
        self.idx2i_ = {i: it for it, i in self.i2idx_.items()}

        docs = [_tokenize(item_text[it]) for it in self.items_]

        # 构建词表 + IDF
        df_counter: Counter = Counter()
        for toks in docs:
            for t in set(toks):
                df_counter[t] += 1
        self.vocab_ = {t: j for j, t in enumerate(sorted(df_counter))}
        n_docs = len(docs)
        idf = np.zeros(len(self.vocab_), dtype=np.float32)
        for t, j in self.vocab_.items():
            idf[j] = np.log((1 + n_docs) / (1 + df_counter[t])) + 1.0
        self.idf_ = idf

        # TF-IDF 矩阵 (n_items, vocab) 并 L2 归一化
        n_i, n_v = len(self.items_), len(self.vocab_)
        tfidf = np.zeros((n_i, n_v), dtype=np.float32)
        for i, toks in enumerate(docs):
            if not toks:
                continue
            tf = Counter(toks)
            inv = 1.0 / len(toks)
            for t, c in tf.items():
                j = self.vocab_.get(t)
                if j is not None:
                    tfidf[i, j] = c * inv * idf[j]
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-9
        self.item_vecs_ = tfidf / norms

        # 用户画像 = 其训练集高分物品向量的加权平均
        self.user_profiles_ = {}
        self.user_seen_ = {}
        for u, g in train.groupby(USER_COL):
            seen = set(g[ITEM_COL].tolist())
            self.user_seen_[u] = seen
            liked = g[g[RATING_COL] >= self.like_threshold]
            if len(liked) == 0:
                liked = g  # 没有高分则用全部历史兜底
            vecs, ws = [], []
            for it, r in liked[[ITEM_COL, RATING_COL]].itertuples(index=False):
                j = self.i2idx_.get(it)
                if j is not None:
                    vecs.append(self.item_vecs_[j])
                    ws.append(float(r))
            if vecs:
                prof = np.average(np.vstack(vecs), axis=0, weights=ws)
                n = np.linalg.norm(prof) + 1e-9
                self.user_profiles_[u] = (prof / n).astype(np.float32)

        self._fitted = True
        return self

    def _get_state(self) -> dict:
        return {"global_mean": self.global_mean_, "items": self.items_,
                "i2idx": self.i2idx_, "idx2i": self.idx2i_,
                "vocab": self.vocab_, "idf": self.idf_,
                "item_vecs": self.item_vecs_,
                "user_profiles": self.user_profiles_,
                "user_seen": self.user_seen_}

    def _set_state(self, state: dict) -> None:
        self.global_mean_ = state["global_mean"]; self.items_ = state["items"]
        self.i2idx_ = state["i2idx"]; self.idx2i_ = state["idx2i"]
        self.vocab_ = state["vocab"]; self.idf_ = state["idf"]
        self.item_vecs_ = state["item_vecs"]
        self.user_profiles_ = state["user_profiles"]
        self.user_seen_ = state["user_seen"]; self._fitted = True

    def _score_user(self, user_id: int) -> np.ndarray:
        prof = self.user_profiles_.get(user_id)
        if prof is None:
            return None
        return self.item_vecs_ @ prof  # 余弦(均已归一化)

    def predict(self, user_id: int, item_id: int) -> float:
        prof = self.user_profiles_.get(user_id)
        iidx = self.i2idx_.get(item_id)
        if prof is None or iidx is None:
            return self.global_mean_
        sim = float(self.item_vecs_[iidx] @ prof)  # [-1, 1]
        # 将相似度映射到评分量纲(1~5),便于参与 RMSE/MAE
        return float(np.clip(self.global_mean_ + sim, 1.0, 5.0))

    def scores_for_items(self, user_id: int, item_ids) -> np.ndarray:
        """向量化:返回该用户对给定 item_ids 的内容相似分(评分量纲)。
        无画像或未知物品回退全局均值。供 Hybrid 批量融合使用。
        """
        mu = self.global_mean_
        out = np.full(len(item_ids), mu, dtype=np.float32)
        prof = self.user_profiles_.get(user_id)
        if prof is None:
            return out
        iidxs = np.array([self.i2idx_.get(int(it), -1) for it in item_ids])
        known = iidxs >= 0
        if known.any():
            sims = self.item_vecs_[iidxs[known]] @ prof
            out[known] = np.clip(mu + sims, 1.0, 5.0)
        return out

    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        scores = self._score_user(user_id)
        if scores is None:
            return []
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
