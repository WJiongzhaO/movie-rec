"""
离线构建 IMDb 元数据缓存(第二数据源融合)
==========================================
独立于 MovieLens(GroupLens) 的另一来源:IMDb 官方公开数据集
(https://datasets.imdbws.com/title.basics.tsv.gz)。两者来源不同、模态不同:
  - MovieLens 1M:用户-物品**评分行为**(协同信号)+ 自带 genres/title
  - IMDb basics :物品侧**权威元数据**(规范化 genres、片长 runtimeMinutes、
                  原始片名 originalTitle、titleType)

本脚本做一次性离线 ETL:
  1. 读 IMDb title.basics.tsv.gz,过滤为电影(movie / tvMovie)。
  2. 按 (规范化标题, 年份) 建索引。
  3. 读 MovieLens movies.dat,规范化标题后与 IMDb 连接(title+year)。
  4. 把 IMDb 的 genres / runtimeMinutes / originalTitle 融进每个 MovieLens item,
     写成 data_raw/imdb_cache.json(供 loader 零网络复现)。

输出缓存结构:{ str(movie_id): {"imdb_genres":..,"runtime":..,"orig_title":..,
                               "tconst":..} }
跑法:python scripts/build_imdb_cache.py
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMDB_GZ = ROOT / "data_raw" / "title.basics.tsv.gz"
ML_MOVIES = ROOT / "data_raw" / "ml-1m" / "movies.dat"
OUT_CACHE = ROOT / "data_raw" / "imdb_cache.json"

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_PAREN_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def norm_title(title: str) -> str:
    """规范化标题用于跨源匹配:去年份后缀、'The/A/An' 后置还原、小写去标点。
    MovieLens: 'Shawshank Redemption, The (1994)' -> imdb 'The Shawshank Redemption'
    """
    t = _PAREN_RE.sub("", str(title)).strip()
    m = re.match(r"^(.*),\s*(The|A|An|La|Le|Les|Il|El)$", t, re.IGNORECASE)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return _NONALNUM.sub(" ", t.lower()).strip()


def load_imdb_index() -> dict:
    """构建 {(norm_title, year): row} 与 {norm_title: row}(年份缺失兜底)。"""
    by_title_year: dict = {}
    by_title: dict = {}
    with gzip.open(IMDB_GZ, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["titleType"] not in ("movie", "tvMovie"):
                continue
            year = r.get("startYear", "\\N")
            genres = r.get("genres", "\\N")
            if genres == "\\N":
                genres = ""
            runtime = r.get("runtimeMinutes", "\\N")
            runtime = "" if runtime == "\\N" else runtime
            for title_field in (r.get("primaryTitle"), r.get("originalTitle")):
                if not title_field:
                    continue
                nt = norm_title(title_field)
                rec = {
                    "tconst": r["tconst"],
                    "imdb_genres": genres,
                    "runtime": runtime,
                    "orig_title": r.get("originalTitle", ""),
                }
                if year != "\\N":
                    by_title_year.setdefault((nt, year), rec)
                by_title.setdefault(nt, rec)
    return by_title_year, by_title


def main() -> None:
    print(f"[1/3] 读取 IMDb 数据集索引 … ({IMDB_GZ.name})")
    by_ty, by_t = load_imdb_index()
    print(f"      IMDb 电影标题索引: {len(by_ty)} 条 (title+year), {len(by_t)} 条 (title)")

    print(f"[2/3] 读取 MovieLens movies.dat 并连接 …")
    cache: dict = {}
    matched_ty = matched_t = total = 0
    with open(ML_MOVIES, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            mid, title, _ml_genres = parts[0], parts[1], parts[2]
            total += 1
            ym = _YEAR_RE.search(title)
            year = ym.group(1) if ym else None
            nt = norm_title(title)
            rec = None
            if year and (nt, year) in by_ty:
                rec = by_ty[(nt, year)]
                matched_ty += 1
            elif nt in by_t:
                rec = by_t[nt]
                matched_t += 1
            if rec:
                cache[str(mid)] = rec

    print(f"[3/3] 写缓存 → {OUT_CACHE}")
    OUT_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    n = len(cache)
    print("--- 融合统计 ---")
    print(f"MovieLens 电影总数        : {total}")
    print(f"title+year 精确命中        : {matched_ty}")
    print(f"title 兜底命中             : {matched_t}")
    print(f"IMDb 元数据覆盖物品数      : {n}  ({n / total:.1%})")
    with_g = sum(1 for v in cache.values() if v["imdb_genres"])
    with_r = sum(1 for v in cache.values() if v["runtime"])
    print(f"  含 IMDb genres           : {with_g}")
    print(f"  含 runtime               : {with_r}")


if __name__ == "__main__":
    main()
