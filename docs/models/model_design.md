# 模型设计文档

> 7 个推荐算法, 统一 `BaseRecommender` 接口, 覆盖协同过滤 / 内容画像 / 混合融合 / 冷启动四大类。

---

## 统一接口 (`base.py`)

所有模型继承 `BaseRecommender`, 实现三个核心方法:

| 方法 | 签名 | 职责 |
|---|---|---|
| `fit(split, dataset)` | → self | 用训练集拟合模型参数 |
| `predict(user_id, item_id)` | → float | 预测评分 (1–5), 用于 RMSE/MAE |
| `recommend(user_id, k)` | → List[int] | Top-K 推荐, 排除已交互物品 |

**新增持久化接口**:

| 方法 | 说明 |
|---|---|
| `_get_state()` → dict | 导出可序列化的模型参数 |
| `_set_state(state)` | 从字典恢复参数 |
| `save(path)` | gzip+pickle 压缩保存 |
| `load(path)` → model | 静态方法, 从文件加载 |

---

## 模型一览

| # | 模型 | 类 | 类别 | 核心思想 |
|---|---|---|---|---|
| 1 | Popularity | `PopularityRecommender` | 基线 | 贝叶斯平滑热门推荐 |
| 2 | ItemCF | `ItemCFRecommender` | 协同过滤 | 物品余弦相似度 + top-N 邻居截断 |
| 3 | SVD | `SVDRecommender` | 协同过滤 | FunkSVD 偏置模型 + mini-batch SGD |
| 4 | ALS | `ALSRecommender` | 协同过滤 | 交替最小二乘闭式解 |
| 5 | Content | `ContentRecommender` | 内容画像 | TF-IDF 物品画像 + 用户偏好加权聚合 |
| 6 | Hybrid | `HybridRecommender` | 混合融合 | SVD + Content min-max 归一化加权 |
| 7 | ColdStart | `ColdStartRecommender` | 冷启动 | Content 画像 + Popularity 兜底双通路 |

---

## 1. Popularity (热门基线)

**算法**: 贝叶斯平滑 `score = (C·m + count·mean) / (C + count)`

- C=5 为平滑强度; m=全局平均分
- 按 score 降序排列所有物品; 推荐时排除用户已交互项
- 作用: 非个性化下限参照——若 CF 还不如热门, 说明有 bug

**超参**: 无 (仅贝叶斯平滑常数 C=5)

---

## 2. ItemCF (物品协同过滤)

**算法**:
1. 构建 user-item 评分矩阵 (n_users × n_items)
2. 列向量余弦相似度矩阵 `sim = normalized.T @ normalized`
3. Top-N 邻居截断: 每个物品只保留 `topk_neighbors` 个最相似邻居, 其余置 0
4. 预测 = 用户已评物品与目标物品相似者的加权平均分
5. 推荐 = 对全物品向量化打分 `scores = sim @ user_vec`

**超参**: `topk_neighbors` (最优 120, 来自网格搜索)

**复杂度**: O(n_items²) 相似度存储, 真实 1M 约 ~110MB

---

## 3. SVD (FunkSVD 矩阵分解)

**模型**: `r̂ = μ + b_u + b_i + p_u · q_i`

**优化**: mini-batch SGD + L2 正则, `np.add.at` 向量化梯度累加 (避免逐样本循环)

**超参**:

| 参数 | 最优值 | 说明 |
|---|---|---|
| n_factors | 50 | 隐因子维度 |
| n_epochs | 20 | 完整训练轮数 |
| lr | 0.005 | 学习率 |
| reg | 0.02 | L2 正则强度 |
| batch_size | 2048 | 小批量大小 |

**关键发现**: 粗筛阶段 n_epochs=8 误选 n_factors=100; 完整训练 (n_epochs=20) 复核后真正最优是 n_factors=50, RMSE 0.890 → 说明小 epoch 低估大模型。

---

## 4. ALS (交替最小二乘) ★新增

**算法**: `min Σ(r - p_u·q_i)² + λ(||p_u||² + ||q_i||²)`

- 固定 Q → 对每用户解 ridge 回归 → 更新 P
- 固定 P → 对每物品解 ridge 回归 → 更新 Q
- 闭式解: `P_u = (Q[I_u]^T Q[I_u] + λI)^(-1) Q[I_u]^T R[I_u]`
- 无需学习率, 收敛稳定, 天然支持并行化

**超参**: `n_factors=50, reg=0.1, n_epochs=15`

**对比价值**: 与 SVD (SGD) 形成优化方法维度的对比基线。

---

## 5. Content (基于内容)

**特征构造**:
- 物品文本 = `genres` (MovieLens ∪ IMDb 并集, `|` 分隔)
- TF-IDF 向量化 (纯 numpy, 无 sklearn 依赖)
- 用户画像 = 高分物品 (rating ≥ 4.0) TF-IDF 向量的加权平均
- 余弦相似度检索

**超参**: `like_threshold=4.0` (高分阈值)

**特点**: 覆盖率最高 (80%), 天然支持冷物品 (只要有 genres 即可被检索)

---

## 6. Hybrid (混合融合)

**融合方式**: 加权式 (weighted)
```
score = α × z(cf_score) + (1-α) × z(content_score)
```
其中 z(·) = min-max 归一化 (消除 SVD 评分量纲与 Content 余弦相似度量纲差异)

**超参**: `alpha=0.75` (CF 主导 + 内容补充, 来自召回率网格搜索调优)

**效果**: RMSE 接近 SVD (0.922), recall 超过纯 SVD, 多样性更好。

---

## 7. ColdStart (冷启动)

**双通路设计**:
| 用户类型 | 路径 | 效果 |
|---|---|---|
| 已知用户 | Content 画像 | 个性化 |
| 新用户 (无历史) | Popularity 兜底 | 热门但非个性化 |

**额外接口**: `recommend_for_preferences(genres, k)` — 用用户选择的类型构造伪画像向量, 在类型多热空间中做余弦检索

**关键指标**: 冷用户服务率 100% vs 纯 CF (SVD/ItemCF) 0%

---

## 设计要点

- **接口优先**: `BaseRecommender` 锁定后, 新增模型只需继承 + 实现 3 个方法 + `_get_state/_set_state`
- **纯 NumPy 实现**: 无深度学习框架依赖, 安装轻量, 1M 数据秒级训练
- **持久化**: gzip+pickle 压缩, 单模型 ~1–5MB, 加载后 `predict` 输出与保存前完全一致
