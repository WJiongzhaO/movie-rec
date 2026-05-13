"""
数据契约 (Data Contract)
========================
定义全项目统一的数据结构。所有模块(模型、评估、界面)只依赖这里的约定,
互不耦合。这是 "Interface First" 原则在数据层的落地。

约定的核心对象是 Dataset:
  - ratings : pd.DataFrame, 列 = [user_id, item_id, rating, timestamp]
  - items   : pd.DataFrame, 列 = [item_id, title, genres, overview, ...]
  - users   : pd.DataFrame, 列 = [user_id, gender, age, occupation]

任何数据来源(真实 MovieLens / TMDB,或 mock 合成数据)最终都被归一化成
这个结构,因此实验框架无需关心数据从哪来。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# 列名常量 —— 全项目统一引用,避免硬编码字符串拼写不一致
USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"
TIME_COL = "timestamp"


@dataclass
class Dataset:
    """统一数据集容器。"""

    ratings: pd.DataFrame
    items: pd.DataFrame
    users: Optional[pd.DataFrame] = None
    name: str = "unnamed"
    meta: dict = field(default_factory=dict)

    # ---- 基础校验:确保任何来源的数据都满足契约 ----
    def validate(self) -> "Dataset":
        required = {USER_COL, ITEM_COL, RATING_COL}
        missing = required - set(self.ratings.columns)
        if missing:
            raise ValueError(f"ratings 缺少必要列: {missing}")
        if ITEM_COL not in self.items.columns:
            raise ValueError(f"items 缺少必要列: {ITEM_COL}")
        return self

    # ---- 常用统计:用于实验记录里的数据摘要 ----
    @property
    def n_users(self) -> int:
        return self.ratings[USER_COL].nunique()

    @property
    def n_items(self) -> int:
        return self.items[ITEM_COL].nunique()

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def density(self) -> float:
        denom = self.n_users * self.n_items
        return self.n_ratings / denom if denom else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_ratings": self.n_ratings,
            "density": round(self.density, 5),
        }


@dataclass
class Split:
    """训练/测试划分结果。"""

    train: pd.DataFrame
    test: pd.DataFrame
    # 冷启动子集:训练集完全未见的用户,用于冷启动分层评估
    cold_user_test: Optional[pd.DataFrame] = None
