"""
IMDb 元数据融合 (第二数据源)
============================
独立于 MovieLens(GroupLens) 的另一来源:IMDb 官方公开数据集
(https://datasets.imdbws.com/)。来源不同、模态不同——MovieLens 提供
用户-物品**评分行为**(协同信号),IMDb 提供物品侧**权威元数据**
(规范化 genres、片长 runtime、原始片名、tconst)。

设计与 tmdb.py 一致,**永不阻塞主流程**:
  1. 若存在离线缓存(data_raw/imdb_cache.json)→ 直接读取合并,零网络、可复现
     (缓存由 scripts/build_imdb_cache.py 通过 title+year 离线连接生成)。
  2. 无缓存 → 保持原样(imdb_genres/runtime 留空)→ content 自动退化为
     MovieLens-only。

融合策略:把 IMDb 的 genres 与 MovieLens 自带 genres **并集**写入物品文本,
runtime 作为数值特征旁路保留。两源 genres 体系不同(IMDb 更细),并集后
content/hybrid 的物品画像信息量显著增加,这正是"多源融合"的价值所在。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import ITEM_COL

_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data_raw" / "imdb_cache.json"


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def enrich_items_with_imdb(
    items: pd.DataFrame,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """把 IMDb 元数据融进 items,返回新 DataFrame(不修改入参)。

    新增/增强列:
      - imdb_genres : IMDb 规范化 genres(逗号分隔)
      - runtime     : 片长(分钟,数值;缺失为 NaN)
      - genres      : MovieLens genres ∪ IMDb genres(用 '|' 连接,去重)
    """
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    cache = _load_cache(cache_path)
    out = items.copy()
    out["imdb_genres"] = ""
    out["runtime"] = pd.NA

    if not cache:
        return out

    def _row_lookup(iid):
        return cache.get(str(iid), {})

    imdb_g, run, fused_g = [], [], []
    for row in out.itertuples(index=False):
        rd = row._asdict()
        rec = _row_lookup(rd[ITEM_COL])
        ig = rec.get("imdb_genres", "") or ""
        rt = rec.get("runtime", "") or ""
        imdb_g.append(ig)
        run.append(int(rt) if str(rt).isdigit() else pd.NA)
        # genres 并集:MovieLens '|' 分隔 + IMDb ',' 分隔
        ml_set = {g.strip() for g in str(rd.get("genres", "")).split("|") if g.strip()}
        imdb_set = {g.strip() for g in ig.split(",") if g.strip()}
        fused = sorted(ml_set | imdb_set)
        fused_g.append("|".join(fused) if fused else str(rd.get("genres", "")))

    out["imdb_genres"] = imdb_g
    out["runtime"] = run
    out["genres"] = fused_g
    return out


def cache_status(cache_path: str | Path | None = None) -> dict:
    """报告 IMDb 融合覆盖情况,写进实验记录以声明第二数据源来源。"""
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    cache = _load_cache(cache_path)
    n = len(cache)
    n_genres = sum(1 for v in cache.values() if v.get("imdb_genres"))
    n_runtime = sum(1 for v in cache.values() if v.get("runtime"))
    return {
        "imdb_cache_path": str(cache_path),
        "imdb_matched_items": n,
        "imdb_with_genres": n_genres,
        "imdb_with_runtime": n_runtime,
    }
