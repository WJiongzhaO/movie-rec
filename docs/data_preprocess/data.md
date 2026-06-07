# 数据文档 ─ MovieLens 1M + IMDb 双源融合

> 本文档覆盖数据源说明、原始格式、预处理步骤、字段字典、质量报告和关键统计。
> 每个字段均可追溯到具体处理代码（`src/data/` 模块）。

---

## 1. 数据源

| 来源 | 机构 | 类型 | 规模 | 获取 |
|---|---|---|---|---|
| **MovieLens 1M** | GroupLens Research | 用户-物品评分行为 | 6040 用户 × 3883 电影 × 1,000,209 评分 | [grouplens.org](https://grouplens.org/datasets/movielens/1m/) |
| **IMDb 公开数据集** | IMDb (datasets.imdbws.com) | 物品侧元数据 | 3476/3883 电影匹配 (89.5%) | 下载 `title.basics.tsv.gz`, 运行 `scripts/build_imdb_cache.py` |

---

## 2. 原始数据格式

### 2.1 ratings.dat

```
分隔符: ::  编码: latin-1  行数: 1,000,209
格式: UserID::MovieID::Rating::Timestamp
示例: 1::1193::5::978300760
```

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| UserID | int | 1–6040 | 用户 ID |
| MovieID | int | 1–3952 (有跳号) | 电影 ID |
| Rating | int | 1–5 | 评分 (整数) |
| Timestamp | int | 956703932 – 1046454590 | Unix 秒 (2000-04 ~ 2003-03) |

### 2.2 movies.dat

```
分隔符: ::  编码: latin-1  行数: 3,883
格式: MovieID::Title (YYYY)::Genres
示例: 1::Toy Story (1995)::Animation|Children's|Comedy
```

| 字段 | 类型 | 说明 |
|---|---|---|
| MovieID | int | 电影 ID |
| Title | str | 标题 + 年份括号 |
| Genres | str | MovieLens 原始类型, `\|` 分隔 (19 种) |

### 2.3 users.dat

```
分隔符: ::  编码: latin-1  行数: 6,040
格式: UserID::Gender::Age::Occupation::Zip-code
示例: 1::F::1::10::48067
```

| 字段 | 类型 | 说明 |
|---|---|---|
| UserID | int | 用户 ID |
| Gender | str | `M` / `F` |
| Age | int | 年龄编码: 1(<18), 18(18–24), 25(25–34), 35(35–44), 45(45–49), 50(50–55), 56(56+) |
| Occupation | int | 职业编码 0–20 |
| Zip-code | str | 美国邮编 |

### 2.4 IMDb 缓存 (imdb_cache.json)

```json
{
  "1": {
    "tconst": "tt0114709",
    "imdb_genres": "Adventure,Animation,Comedy",
    "runtime": "81",
    "orig_title": "Toy Story"
  }
}
```

---

## 3. 预处理管道

| 步骤 | 位置 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| **1. 文件解析** | `loader.py:49-55` | `.dat` 文件 | DataFrame | `::` 分隔, latin-1 解码 |
| **2. 列名映射** | `loader.py:49-55` | 原始列名 | 标准化列名 | UserID→user_id, MovieID→item_id … |
| **3. 年份提取** | `loader.py:64-66` | title 列 | release_year | regex `\((\d{4})\)` |
| **4. IMDb 融合** | `imdb.py:40-79` | movies + imdb_cache | genres 并集 + runtime | ML genres ∪ IMDb genres, 去重排序 |
| **5. 契约校验** | `schema.py:40-48` | Dataset | Dataset | 必含列检查 |
| **6. 质量报告** | `cleaning.py:114-137` | Dataset | meta["quality"] | 去重/缺失值/低质量标记 |
| **7. 特征工程** | `features.py` | Dataset | 类型矩阵/用户特征/统计量 | 多热编码/归一化/统计 |
| **8. 数据划分** | `loader.py:101-212` | Dataset | Split | 时间留出 / 冷启动分层 |

### 3.1 数据清洗详情 (`cleaning.py`)

| 操作 | 函数 | 实际结果 |
|---|---|---|
| 评分去重 | `deduplicate_ratings()` | 0 条重复 (MovieLens 1M 原始质量高) |
| 缺失值统计 | `missing_values_report()` | ratings: 0; items: runtime 421 条 (10.8%); users: 0 |
| 低活跃度标记 | `flag_low_quality()` | 0 低活跃用户; 203 个冷门物品 (≤3 评分) |

### 3.2 特征工程详情 (`features.py`)

| 特征 | 函数 | 维度 | 说明 |
|---|---|---|---|
| 类型多热矩阵 | `build_genre_matrix()` | 3883×23 | 23 种类型 (含 IMDb 引入的 Family/Biography 等) |
| 用户画像 | `build_user_features()` | 6040×12 | gender_M/F, age_normalized, age_bucket_×7, occupation |
| 物品统计 | `build_item_stats()` | 3706×6 | rating_count, mean, std (含归一化) |
| 用户统计 | `build_user_stats()` | 6040×6 | rating_count, mean, std (含归一化) |

---

## 4. 最终 Dataset 字段字典

### 4.1 ratings 表

| 字段 | 类型 | 来源 | 非空率 |
|---|---|---|---|
| `user_id` | int | ratings.dat→UserID | 100% |
| `item_id` | int | ratings.dat→MovieID | 100% |
| `rating` | int | ratings.dat→Rating | 100% |
| `timestamp` | int | ratings.dat→Timestamp | 100% |

### 4.2 items 表

| 字段 | 类型 | 来源 | 非空率 |
|---|---|---|---|
| `item_id` | int | movies.dat→MovieID | 100% |
| `title` | str | movies.dat→Title | 100% |
| `genres` | str | **ML ∪ IMDb 并集** | 100% |
| `release_year` | float | title regex | 100% |
| `imdb_genres` | str | imdb_cache.json | 89.3% |
| `runtime` | int/NA | imdb_cache.json | 89.2% |

### 4.3 users 表

| 字段 | 类型 | 来源 | 非空率 |
|---|---|---|---|
| `user_id` | int | users.dat→UserID | 100% |
| `gender` | str | users.dat→Gender | 100% |
| `age` | int | users.dat→Age | 100% |
| `occupation` | int | users.dat→Occupation | 100% |
| `zip_code` | str | users.dat→Zip-code | 100% |

---

## 5. 关键统计一览

| 维度 | 数值 |
|---|---|
| 总用户数 | 6,040 |
| 总电影数 | 3,883 (活跃评分: 3,706) |
| 总评分数 | 1,000,209 |
| 评分矩阵密度 | 4.47% |
| 评分均值 / 标准差 | 3.58 / 1.12 |
| 评分时间范围 | 2000-04-26 ~ 2003-03-01 |
| 电影年代范围 | 1919–2000 (主力: 1990s 占 58.8%) |
| 每用户评分数 | mean=165.6, median=96, min=20, max=2314 |
| 每物品评分数 | mean=269.9, median=124, min=1, max=3428 |
| IMDb 匹配率 | 89.5% (3476/3883) |
| IMDb genres 覆盖率 | 89.3% |
| IMDb runtime 覆盖率 | 89.2% (mean=105min) |
| 重复评分 | 0 条 |
| 冷门物品 (≤3 评分) | 203 个 |
| 男性用户占比 | 71.7% |
| 主力年龄段 | 25–34 岁 (34.7%) |
