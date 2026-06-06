# 实验对比 & 超参调优

> 覆盖对比实验设计、超参数优化（网格搜索）、结果可复现。

---

## 设计思路

与旧版"实验注册表 + leaderboard"不同，当前采用**纯函数式、按需运行**的轻量设计：

- 不维护全局实验注册表（按需从 YAML config 运行）
- 每次实验的结果以 dict 形式在内存中传递
- 网格搜索直接复用 `run_from_config()`，每个组合自动评估

---

## 文件

| 文件 | 职责 |
|---|---|
| `src/experiment/runner.py` | `run_from_config(config)` → `{model, params, metrics, elapsed_s}` |
| `src/experiment/compare.py` | `compare(run_a, run_b)` → DataFrame (delta / pct / judgment) |
| `src/experiment/grid.py` | `grid_search_from_config(path)` → coarse→refine 全流程 |

---

## 1. 单次实验 (`runner.py`)

```python
from src.experiment.runner import run_from_config

result = run_from_config("configs/svd_ml1m.yaml")
# → {"model": "svd", "params": {...}, "metrics": {...}, "train_s": 14.4}
```

**配置结构** (`configs/*.yaml`):

```yaml
data:
  source: movielens        # 固定为真实数据
split:
  test_ratio: 0.2          # 测试集比例
  cold_user_ratio: 0.1     # (可选) 冷用户比例
model:
  name: svd                # 模型名
  params:                  # 模型超参
    n_factors: 50
    n_epochs: 20
    lr: 0.005
    reg: 0.02
eval:
  k: 10                    # Top-N K 值
  relevance_threshold: 4.0 # 相关性阈值
note: "实验说明"
```

**CLI**:
```bash
python main.py run --config configs/svd_ml1m.yaml
```

---

## 2. 对比实验 (`compare.py`)

```python
from src.experiment.compare import compare

df = compare(run_a, run_b)
# DataFrame: metric | baseline | target | delta | delta_pct | judgment
```

输出示例 (SVD vs ALS):

| metric | baseline | target | delta | delta_pct | judgment |
|---|---|---|---|---|---|
| rmse | 0.890 | 1.211 | +0.321 | +36.1% | ↓ worse |
| recall@10 | 0.028 | 0.001 | -0.027 | -95.7% | ↓ worse |
| coverage | 0.168 | 0.344 | +0.175 | +104.0% | ↑ better |
| novelty | 3.49 | 6.30 | +2.81 | +80.5% | ↑ better |

- 自动识别指标方向: rmse/mae 越小越好; 其余越大越好
- 诊断字段 (`eval_users` 等) 标注为 `(诊断)` 不参与优劣判定

**CLI**:
```bash
python main.py compare configs/svd_ml1m.yaml configs/als_ml1m.yaml
```

---

## 3. 超参网格搜索 (`grid.py`)

**两阶段策略**: 粗筛 (宽区间大步长) → 细化 (围绕最优点插值)

```python
from src.experiment.grid import grid_search_from_config

result = grid_search_from_config("configs/als_grid_ml1m.yaml")
# → {"coarse": DataFrame, "coarse_best": {...},
#    "refine": DataFrame, "refine_best": {...}}
```

**网格配置** (`*_grid_ml1m.yaml`):

```yaml
# (继承普通 config 结构)
data: ...
split: ...
model: {name: svd, params: {lr: 0.005, seed: 42}}
eval: ...

# 网格搜索特有字段
optimize: rmse                    # 排名指标

fixed_params:                     # 所有组合共享 (粗筛省时)
  n_epochs: 8

grid:                             # 候选超参网格
  n_factors: [20, 50, 100]
  reg: [0.01, 0.02, 0.05]

refine:                           # 细化阶段
  enabled: true
  n_between: 1                    # 相邻候选间插值点数
```

**CLI**:
```bash
python main.py grid --config configs/svd_grid_ml1m.yaml
```

### 已完成的调优结果

| 模型 | 调优对象 | 粗筛最优 | 细化最优 | 备注 |
|---|---|---|---|---|
| SVD | n_factors × reg | n_factors=100, reg=0.02 | n_factors=100, reg=0.015 | **注意**: n_epochs=8 粗筛; 完整训练 (n_epochs=20) 真正最优为 n_factors=50, reg=0.02 (RMSE 0.890) |
| ItemCF | topk_neighbors | 80 | **120** | RMSE 1.008 |
| Hybrid | alpha | — | **0.75** | Recall@10 最优; CF 主导 + 内容补充 |
| ALS | n_factors × reg | 待运行 | 待运行 | 首次调优 |

### 关键方法论教训

**SVD 粗筛 bias**: `n_epochs=8` 时粗筛误选 `n_factors=100`；完整训练 (`n_epochs=20`) 复核后真正最优为 `n_factors=50`。说明**小 epoch 低估大模型**，网格搜索的固定参数选择会影响最终结论，应以完整训练复核。

**ItemCF 超参修复**: 调参前发现 `topk_neighbors` 声明但从未在代码中使用（所有取值指标相同）。修复邻居截断逻辑后，该超参才开始生效——"检测方法有缺陷时先修环境"。
