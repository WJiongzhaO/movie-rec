# 评估体系设计

> 三层评估框架 + 统计显著性检验 + 错误分析 + 可视化。

---

## 1. 三层指标体系 (`metrics.py`)

### L1 评分预测
| 指标 | 公式 | 方向 |
|---|---|---|
| RMSE | `sqrt(mean((y - y_hat)^2))` | ↓ |
| MAE | `mean(\|y - y_hat\|)` | ↓ |

### L2 Top-N 排序
| 指标 | 公式 | 方向 |
|---|---|---|
| Precision@K | `hits / K` | ↑ |
| Recall@K | `hits / \|relevant\|` | ↑ |
| HitRate@K | `1 if any hit else 0` | ↑ |
| NDCG@K | `DCG / IDCG` (对数折扣累加增益) | ↑ |
| MAP@K | `mean(avg_precision)` | ↑ |

### L3 业务指标
| 指标 | 计算 | 方向 |
|---|---|---|
| Coverage | 被推荐物品种类 / 总物品数 | ↑ |
| Diversity | 推荐列表内 Jaccard 距离均值 | ↑ |
| Novelty | 推荐物品 self-information 均值 | ↑ |

---

## 2. 评估器 (`evaluator.py`)

```
Evaluator(k=10, relevance_threshold=4.0)
  ├── evaluate(model, split, dataset) → {metric: value, ...}
  │     L1: 全测试集逐条预测
  │     L2: 有相关物品的用户 → 推荐 → Top-N 指标
  │     L3: 聚合全量推荐列表
  │     诊断: eval_users, eval_empty_rec, eval_rel_size_mean
  │
  └── evaluate_cold_users(model, split, dataset) → {cold_metric: value, ...}
        仅在 cold_user_test 上评估 Top-N
        新增 cold_serve_rate (模型能服务冷用户的比例)
```

**可靠性护栏**: `eval_rel_size_mean` 反映每用户平均相关物品数。真实 1M 约 17.3，Mock 约 1.2。数值过小会导致 precision@k 结构性压低、recall 与 hitrate 高度重合。

---

## 3. 统计显著性检验 (`significance.py`)

| 检验 | 函数 | 用途 |
|---|---|---|
| 配对 t 检验 | `paired_ttest()` | 两模型在同一用户上的指标差异是否为 0 |
| Bootstrap CI | `bootstrap_ci()` | 任意指标的置信区间 (无需正态假设) |
| Cohen's d | `cohens_d()` | 效应量 (\|d\|<0.2 可忽略; >0.8 大效应) |
| 一站式对比 | `compare_models_per_user()` | t-test + bootstrap + effect size |

**典型用法**:
```python
from src.evaluation.significance import compare_models_per_user
from src.evaluation.error_analysis import per_user_rmse

result = compare_models_per_user(
    per_user_rmse(svd, split),
    per_user_rmse(itemcf, split),
)
# → {ttest: {p_value, significant}, bootstrap_a/b, cohens_d}
```

**实测示例 (SVD vs ItemCF)**:
- t = -36.9, p ≈ 0 — SVD 的每用户 RMSE 显著优于 ItemCF
- SVD RMSE 95% CI: [0.876, 0.890]
- ItemCF RMSE 95% CI: [0.976, 0.993]
- Cohen's d = 0.34 (小效应) — 显著但效应量不大

---

## 4. 错误分析 (`error_analysis.py`)

### 4.1 每用户指标
- `per_user_rmse(model, split)` → `{user_id: rmse}`
- `per_user_recall(model, split, dataset, k)` → `{user_id: recall@k}`
- 用于统计检验的配对输入

### 4.2 用户活跃度分层
- `by_user_activity(model, split, dataset, n_buckets=4)`
- 按训练集评分数将用户均匀分桶，每桶独立评估 RMSE + Recall

**实测发现 (SVD)**:
| 桶 | 平均评分数 | RMSE | Recall@10 |
|---|---|---|---|
| Q1 (低活跃) | ~45 | 0.968 | 0.027 |
| Q2 | ~80 | 0.937 | 0.029 |
| Q3 | ~130 | 0.911 | 0.027 |
| Q4 (高活跃) | ~300 | **0.869** | 0.029 |

结论: SVD 对高活跃用户更准 (RMSE 下降 10%)，但 Recall 几乎不变。

### 4.3 物品热度分层
- `by_item_popularity(model, split, dataset, n_buckets=4)`
- 按训练集被评分次数将物品分桶

**实测发现 (SVD)**:
| 桶 | 平均评分数 | HitRate@10 |
|---|---|---|
| Q1 (冷门) | ~8 | 0.000 |
| Q2 | ~30 | 0.007 |
| Q3 | ~90 | 0.023 |
| Q4 (热门) | ~300 | **0.271** |

结论: SVD 对冷门物品几乎完全失效——这正是需要 Content/Hybrid 等补充的原因。

### 4.4 评分误差分布
- `rating_error_distribution(model, split)`
- 过估比例 / 低估比例 / 精确比例 (\|error\|<0.5)

**实测发现 (SVD)**: 过估 49% / 低估 51% / 精确 44%。模型对极端评分 (1/5) 的误差更大。

---

## 5. 可视化 (`visualize.py`)

| 图表 | 函数 | 用途 |
|---|---|---|
| 模型对比 | `plot_model_comparison()` | 三联柱状图 (评分/排序/业务) |
| 冷启动对比 | `plot_coldstart_comparison()` | 服务率 + 命中率 |
| 多 K 曲线 | `plot_multik_curves()` | Recall/NDCG/HitRate @5,10,15,20 |
| 多模型叠加 | `plot_multik_models()` | 多模型同一坐标系对比 |
| 误差分布 | `plot_error_distribution()` | 直方图 + 按真实评分误差 |
| 活跃度分层 | `plot_activity_breakdown()` | RMSE + Recall 双轴柱状 |
| 热度分层 | `plot_popularity_breakdown()` | 冷→热 命中率递减 |

---

## 6. 关键发现

1. **显著性 ≠ 效应量**: SVD 的 RMSE 显著优于 ItemCF (p≈0)，但 Cohen's d=0.34 仅为小效应。
2. **冷物品是薄弱环节**: SVD 对冷门物品 (≤30 评分) 的 HitRate 接近 0，Content 模型可覆盖该缺口。
3. **活跃用户受益更多**: RMSE 随用户评分数单调下降 (0.97 → 0.87)，说明更多行为数据 → 更准预测。
4. **误差分布对称**: 过估与低估几乎平衡，无系统性偏差。
