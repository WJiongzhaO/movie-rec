"""
特征工程模块
============
从原始 Dataset 中抽取可供模型直接使用的特征矩阵和统计量。

功能:
  - build_genre_matrix(): 多热编码 (n_items × n_genres)
  - build_user_features(): 用户画像特征 (性别/年龄/职业编码)
  - build_item_features(): 物品统计特征 (评分次数/均值/方差)
  - build_user_stats(): 用户行为统计
  - build_all_features(): 一键产出全部特征
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .schema import Dataset, ITEM_COL, RATING_COL, USER_COL


# ═══════════════════════════════════════════════════════════════════
# 类型多热编码
# ═══════════════════════════════════════════════════════════════════

def build_genre_matrix(items: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    """从 items 表的 genres 列构建多热编码矩阵。

    参数: items — 必须含 item_id 和 genres 列 (genres 以 '|' 分隔)
    返回: (矩阵 shape=(n_items, n_genres), 类型名列表)
    """
    # 收集全部类型名
    all_genres: set[str] = set()
    for gs in items["genres"]:
        for g in str(gs).split("|"):
            g = g.strip()
            if g:
                all_genres.add(g)
    genre_list = sorted(all_genres)
    g2idx = {g: i for i, g in enumerate(genre_list)}
    n_items, n_genres = len(items), len(genre_list)
    mat = np.zeros((n_items, n_genres), dtype=np.float32)

    for row_i, (_, row) in enumerate(items.iterrows()):
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g in g2idx:
                mat[row_i, g2idx[g]] = 1.0
    return mat, genre_list


# ═══════════════════════════════════════════════════════════════════
# 用户画像特征
# ═══════════════════════════════════════════════════════════════════

def build_user_features(users: pd.DataFrame) -> pd.DataFrame:
    """将原始用户表转化为数值特征。

    输入列:  user_id, gender, age, occupation
    输出列:  user_id, gender_M, gender_F, age_normalized,
             occupation (保留原始, 后续可 one-hot),
             age_group_1~7 (年龄段 one-hot)
    """
    df = users.copy()
    # 性别 → 二值
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["gender_F"] = (df["gender"] == "F").astype(int)

    # 年龄 → 归一化 (映射到 0~1)
    age_raw = df["age"].astype(float)
    df["age_normalized"] = (age_raw - age_raw.min()) / (age_raw.max() - age_raw.min() + 1e-9)

    # 年龄段 one-hot (7 组)
    age_values = sorted(df["age"].unique())
    for av in age_values:
        df[f"age_bucket_{int(av)}"] = (df["age"] == av).astype(int)

    return df[[USER_COL, "gender_M", "gender_F", "age_normalized", "occupation"]
              + [f"age_bucket_{int(av)}" for av in age_values]]


# ═══════════════════════════════════════════════════════════════════
# 物品 / 用户 统计特征
# ═══════════════════════════════════════════════════════════════════

def build_item_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """每个物品的评分统计。

    列: item_id, item_rating_count, item_rating_mean, item_rating_std
    """
    agg = ratings.groupby(ITEM_COL)[RATING_COL].agg(["count", "mean", "std"]).reset_index()
    agg.columns = [ITEM_COL, "item_rating_count", "item_rating_mean", "item_rating_std"]
    agg["item_rating_std"] = agg["item_rating_std"].fillna(0.0)
    # 归一化到 0~1
    for col in ["item_rating_count", "item_rating_mean"]:
        lo, hi = agg[col].min(), agg[col].max()
        if hi > lo:
            agg[f"{col}_norm"] = (agg[col] - lo) / (hi - lo)
        else:
            agg[f"{col}_norm"] = 0.0
    return agg


def build_user_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """每个用户的评分行为统计。

    列: user_id, user_rating_count, user_rating_mean, user_rating_std
    """
    agg = ratings.groupby(USER_COL)[RATING_COL].agg(["count", "mean", "std"]).reset_index()
    agg.columns = [USER_COL, "user_rating_count", "user_rating_mean", "user_rating_std"]
    agg["user_rating_std"] = agg["user_rating_std"].fillna(0.0)
    for col in ["user_rating_count", "user_rating_mean"]:
        lo, hi = agg[col].min(), agg[col].max()
        if hi > lo:
            agg[f"{col}_norm"] = (agg[col] - lo) / (hi - lo)
        else:
            agg[f"{col}_norm"] = 0.0
    return agg


# ═══════════════════════════════════════════════════════════════════
# 一键产出
# ═══════════════════════════════════════════════════════════════════

def build_all_features(dataset: Dataset) -> dict:
    """一键从 Dataset 中抽取全部特征, 返回 dict 供下游使用。

    返回:
      {
        "genre_matrix":    np.ndarray (n_items × n_genres),
        "genre_names":     list[str],
        "user_features":   pd.DataFrame (user_id + 特征列),
        "item_stats":      pd.DataFrame (item_id + 统计列),
        "user_stats":      pd.DataFrame (user_id + 统计列),
      }
    """
    return {
        "genre_matrix":   build_genre_matrix(dataset.items)[0],
        "genre_names":    build_genre_matrix(dataset.items)[1],
        "user_features":  build_user_features(dataset.users),
        "item_stats":     build_item_stats(dataset.ratings),
        "user_stats":     build_user_stats(dataset.ratings),
    }
