"""
推荐模型统一接口 (BaseRecommender)
==================================
所有算法(协同过滤 / 基于内容 / 冷启动 / 混合)都实现这一接口,
实验框架才能 "对任意模型一视同仁地训练和评估"。这是算法系统性对比
能够成立的前提。

接口约定:
  - fit(split)            : 用训练集拟合模型
  - predict(user, item)   : 预测单个评分(用于 RMSE/MAE)
  - recommend(user, k)    : 返回 Top-K 物品列表(用于 Recall@K/NDCG@K 等)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..data.schema import Dataset, Split


class BaseRecommender(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self._fitted = False

    @abstractmethod
    def fit(self, split: Split, dataset: Dataset) -> "BaseRecommender":
        """训练。dataset 提供 items/users 等内容信息(content-based 需要)。"""
        ...

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """预测评分。未知 user/item 应返回一个合理的回退值(如全局均值)。"""
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10) -> List[int]:
        """返回 Top-K 物品 id 列表(已排除该用户训练集中已交互过的物品)。"""
        ...

    # ── 持久化接口 ──
    def _get_state(self) -> dict:
        """子类重写: 返回可序列化的模型参数字典。"""
        return {}

    def _set_state(self, state: dict) -> None:
        """子类重写: 从字典恢复模型参数。"""
        pass

    def save(self, path: str) -> None:
        """保存模型到磁盘。"""
        import pickle
        import gzip
        data = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": self.params,
            "state": self._get_state(),
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=5)

    @staticmethod
    def load(path: str) -> "BaseRecommender":
        """从磁盘加载模型。"""
        import pickle
        import gzip
        import importlib
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        mod = importlib.import_module(data["module"])
        cls = getattr(mod, data["class"])
        model = cls(**data["params"])
        model._set_state(data["state"])
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
