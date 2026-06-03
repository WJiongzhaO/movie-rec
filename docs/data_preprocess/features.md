# 特征工程文档

> 本文档说明从原始 Dataset 中抽取的工程特征，覆盖特征的维度、含义、生成方式。

---

## 特征总览

| 特征 | 维度 | 类型 | 生成函数 | 用途 |
|---|---|---|---|---|
| `genre_matrix` | 3883 × 23 | float32 | `build_genre_matrix()` | Content / ColdStart 的内容侧输入 |
| `user_features` | 6040 × 12 | float64 | `build_user_features()` | 用户画像 (可用于未来模型扩展) |
| `item_stats` | 3706 × 6 | float64 | `build_item_stats()` | 物品热度 / 质量信号 |
| `user_stats` | 6040 × 6 | float64 | `build_user_stats()` | 用户活跃度信号 |

---

## 1. 类型多热矩阵 (`genre_matrix`)

**文件**: `src/data/features.py → build_genre_matrix()`

**输入**: `items.genres` 列 (`|` 分隔的字符串)

**处理**:
1. 扫描全部物品的 genres 字符串，收集所有唯一类型名
2. 排序形成 `genre_list` (23 种)
3. 每行对应一个物品，若该物品具有某类型则置 1

**输出**: `(np.ndarray[3883, 23], List[str])`

**23 种类型** (含 IMDb 融合后引入的 Family / Biography / Music / History / Sport):

```
Action, Adventure, Animation, Biography, Children's, Comedy, Crime,
Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror,
Music, Musical, Mystery, Romance, Sci-Fi, Sport, Thriller, War, Western
```

**当前用途**:
- `ContentRecommender`: 拼接 genres 字符串构建 TF-IDF 文本特征
- `ColdStartRecommender.recommend_for_preferences()`: 用用户选择的类型查询向量 → 余弦检索

---

## 2. 用户画像特征 (`user_features`)

**文件**: `src/data/features.py → build_user_features()`

**输入**: `users` 表

| 输出列 | 来源 | 编码方式 | 含义 |
|---|---|---|---|
| `user_id` | users.user_id | 原样 | 主键 |
| `gender_M` | users.gender | `M` → 1, 其他 → 0 | 男性二值 |
| `gender_F` | users.gender | `F` → 1, 其他 → 0 | 女性二值 |
| `age_normalized` | users.age | `(x-min)/(max-min)` | 年龄归一化到 [0, 1] |
| `occupation` | users.occupation | 原样 (0–20) | 职业编码 (可后续 one-hot) |
| `age_bucket_1` ~ `age_bucket_56` | users.age | one-hot | 7 个年龄段: 1, 18, 25, 35, 45, 50, 56 |

**年龄编码 → 实际含义**:

| 编码值 | 含义 | 人数 | 占比 |
|---|---|---|---|
| 1 | < 18 岁 | 222 | 3.7% |
| 18 | 18–24 岁 | 1,103 | 18.3% |
| 25 | 25–34 岁 | 2,096 | 34.7% |
| 35 | 35–44 岁 | 1,193 | 19.8% |
| 45 | 45–49 岁 | 550 | 9.1% |
| 50 | 50–55 岁 | 496 | 8.2% |
| 56 | 56+ 岁 | 380 | 6.3% |

---

## 3. 物品统计特征 (`item_stats`)

**文件**: `src/data/features.py → build_item_stats()`

**输入**: `ratings` 表

| 输出列 | 计算方式 | 含义 |
|---|---|---|
| `item_id` | 原样 | 主键 |
| `item_rating_count` | `count(rating)` | 被评分次数 (热度) |
| `item_rating_mean` | `mean(rating)` | 平均评分 (质量) |
| `item_rating_std` | `std(rating)`, NA→0 | 评分分歧度 |
| `item_rating_count_norm` | min-max 归一化 | 热度归一化 |
| `item_rating_mean_norm` | min-max 归一化 | 质量归一化 |

**统计分布**:

| 统计 | mean | median | min | max |
|---|---|---|---|---|
| 评分人数 | 269.9 | 124 | 1 | 3428 |
| 平均分 | 3.58 | 3.62 | 0.67 | 5.0 |
| 评分标准差 | 0.93 | 0.94 | 0 | 2.0 |

---

## 4. 用户行为统计 (`user_stats`)

**文件**: `src/data/features.py → build_user_stats()`

**输入**: `ratings` 表

| 输出列 | 计算方式 | 含义 |
|---|---|---|
| `user_id` | 原样 | 主键 |
| `user_rating_count` | `count(rating)` | 评分数 (活跃度) |
| `user_rating_mean` | `mean(rating)` | 平均给分 (宽松/严格) |
| `user_rating_std` | `std(rating)`, NA→0 | 评分区分度 |
| `user_rating_count_norm` | min-max 归一化 | 活跃度归一化 |
| `user_rating_mean_norm` | min-max 归一化 | 给分倾向归一化 |

**统计分布**:

| 统计 | mean | median | min | max |
|---|---|---|---|---|
| 评分数 | 165.6 | 96 | 20 | 2314 |
| 平均给分 | 3.64 | 3.64 | 1.0 | 5.0 |
| 给分标准差 | 1.05 | 1.05 | 0 | 2.0 |

---

## 使用方式

```python
from src.data.features import build_all_features
from src.data.loader import load_data

ds = load_data()
feats = build_all_features(ds)

# 类型矩阵
print(feats["genre_matrix"].shape)    # (3883, 23)

# 用户画像
print(feats["user_features"].columns) # 12 列

# 物品/用户统计 (已归一化, 可直接拼接)
print(feats["item_stats"].columns)    # 6 列
print(feats["user_stats"].columns)    # 6 列
```
