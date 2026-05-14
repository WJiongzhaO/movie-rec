"""
数据加载与划分
==============
- load_data(): 加载 MovieLens 1M + IMDb 融合数据,产出统一 Dataset。
- train_test_split_by_time: 按时间留出法划分训练/测试集。
- train_test_split_with_cold_users: 带冷启动子集的划分。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    Dataset,
    ITEM_COL,
    RATING_COL,
    Split,
    TIME_COL,
    USER_COL,
)

# 真实数据默认目录(相对项目根)
_DEFAULT_ML1M_DIR = Path(__file__).resolve().parents[2] / "data_raw" / "ml-1m"


def load_data(data_dir: str | Path | None = None) -> Dataset:
    """加载 MovieLens 1M 并融合 IMDb 元数据,归一化为统一 Dataset。

    MovieLens 1M (GroupLens):
      - ratings.dat: UserID::MovieID::Rating::Timestamp
      - movies.dat : MovieID::Title::Genres
      - users.dat  : UserID::Gender::Age::Occupation::Zip-code

    IMDb 增强(第二数据源,独立于 GroupLens):
      - 通过离线缓存(data_raw/imdb_cache.json)按 title+year 连接,
        将 IMDb 规范化 genres 与 MovieLens genres 取并集,显著增强物品画像;
      - 覆盖约 89.5% 的电影,无缓存时自动退化为 MovieLens-only。
      - 缓存由 scripts/build_imdb_cache.py 一次性离线生成。
    """
    data_dir = Path(data_dir) if data_dir else _DEFAULT_ML1M_DIR
    if not (data_dir / "ratings.dat").exists():
        raise FileNotFoundError(
            f"未找到 MovieLens 1M 数据于 {data_dir}。"
            "请先下载 https://files.grouplens.org/datasets/movielens/ml-1m.zip 并解压。"
        )

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=[USER_COL, ITEM_COL, RATING_COL, TIME_COL],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        names=[ITEM_COL, "title", "genres"],
        encoding="latin-1",
    )
    movies["release_year"] = (
        movies["title"].str.extract(r"\((\d{4})\)").astype("float")
    )

    # ---- 第二数据源融合:IMDb 官方元数据(独立于 GroupLens) ----
    # 通过离线缓存(title+year 连接)合并 IMDb 规范化 genres / runtime,
    # 与 MovieLens 自带 genres 取并集,显著增强内容画像;无缓存则原样退化。
    imdb_meta: dict = {"imdb_used": False}
    from .imdb import cache_status as imdb_cache_status
    from .imdb import enrich_items_with_imdb

    movies = enrich_items_with_imdb(movies)
    istatus = imdb_cache_status()
    imdb_meta = {
        "imdb_used": istatus["imdb_matched_items"] > 0,
        **istatus,
    }

    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        names=[USER_COL, "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    ds = Dataset(
        ratings=ratings,
        items=movies,
        users=users,
        name="movielens-1m",
        meta={"source_dir": str(data_dir), **imdb_meta},
    ).validate()

    # ---- 数据质量报告 (附在 meta 中) ----
    from .cleaning import comprehensive_quality_report
    quality = comprehensive_quality_report(ds)
    ds.meta["quality"] = quality

    return ds


def train_test_split_by_time(
    dataset: Dataset, test_ratio: float = 0.2
) -> Split:
    """按时间戳留出法:每个用户最近的一部分评分作为测试集。
    更贴近推荐系统真实评估场景(用过去预测未来)。
    """
    df = dataset.ratings.sort_values([USER_COL, TIME_COL])
    train_parts, test_parts = [], []
    for _, g in df.groupby(USER_COL):
        n_test = max(1, int(len(g) * test_ratio))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return Split(train=train, test=test)


def train_test_split_with_cold_users(
    dataset: Dataset,
    test_ratio: float = 0.2,
    cold_user_ratio: float = 0.1,
    seed: int = 42,
) -> Split:
    """带冷启动子集的划分,支撑"冷启动 vs 老用户"分层评估。

    做法:
      1. 随机抽取 cold_user_ratio 比例的用户作为"冷用户"——他们的**全部**评分
         都不进训练集,行为对模型完全不可见(模拟全新用户)。
      2. 其余"老用户"走常规按时间留出(train / test)。

    返回 Split(train, test, cold_user_test=冷用户全部评分)。
    老用户指标走 test,冷启动指标走 cold_user_test,分层对比即误差分析素材。
    """
    rng = np.random.default_rng(seed)
    all_users = dataset.ratings[USER_COL].unique()
    n_cold = max(1, int(len(all_users) * cold_user_ratio))
    cold_users = set(rng.choice(all_users, size=n_cold, replace=False).tolist())

    df = dataset.ratings.sort_values([USER_COL, TIME_COL])
    warm = df[~df[USER_COL].isin(cold_users)]
    cold = df[df[USER_COL].isin(cold_users)]

    # 老用户:常规按时间留出
    train_parts, test_parts = [], []
    for _, g in warm.groupby(USER_COL):
        n_test = max(1, int(len(g) * test_ratio))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return Split(
        train=train,
        test=test,
        cold_user_test=cold.reset_index(drop=True),
    )


def train_test_split_with_cold_items(
    dataset: Dataset,
    test_ratio: float = 0.2,
    cold_item_ratio: float = 0.05,
    seed: int = 42,
) -> Split:
    """带新物品冷启动子集的划分。

    做法:
      1. 随机抽取 cold_item_ratio 比例的低评分次数物品 (≤5条) 作为"冷物品"——
         其**全部评分**不进训练集。
      2. 其余"热物品"走常规按时间留出 (train / test)。
      3. 冷物品的评分全部进入 cold_item_test。

    返回 Split(train, test, cold_item_test=冷物品全部评分)。
    用于评估 content-based 模型在新物品上的冷启动能力。
    """
    rng = np.random.default_rng(seed)
    # 找出低评分次数的物品 (≤5 条评分的冷门物品)
    item_counts = dataset.ratings.groupby(ITEM_COL).size()
    cold_candidates = item_counts[item_counts <= 5]
    if len(cold_candidates) == 0:
        # 如果没有低评分物品, 按比例随机抽取
        all_items = dataset.ratings[ITEM_COL].unique()
        n_cold = max(1, int(len(all_items) * cold_item_ratio))
        cold_items = set(rng.choice(all_items, size=n_cold, replace=False).tolist())
    else:
        n_cold = max(1, int(len(cold_candidates) * cold_item_ratio))
        if n_cold < len(cold_candidates):
            cold_items = set(rng.choice(cold_candidates.index, size=n_cold, replace=False).tolist())
        else:
            cold_items = set(cold_candidates.index.tolist())

    df = dataset.ratings.sort_values([USER_COL, TIME_COL])
    warm = df[~df[ITEM_COL].isin(cold_items)]
    cold = df[df[ITEM_COL].isin(cold_items)]

    # 热物品: 按时间留出
    train_parts, test_parts = [], []
    for _, g in warm.groupby(USER_COL):
        n_test = max(1, int(len(g) * test_ratio))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return Split(
        train=train,
        test=test,
        cold_user_test=cold.reset_index(drop=True),  # 冷物品评分用作 cold_item_test
    )
