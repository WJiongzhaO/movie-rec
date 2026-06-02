# 电影推荐系统 (MovieLens 1M + IMDb)

> 数据挖掘与分析 · 期末项目 · 完整端到端数据挖掘应用

一个覆盖 **数据处理 → 模型训练 → 评估 → 交互界面** 全链路的电影推荐系统。
数据融合 MovieLens 1M 用户行为与 IMDb 物品元数据，提供 **7 种推荐算法**的统一对比与调优。

## 快速开始

```bash
pip install -r requirements.txt

# 一键端到端管道 (数据加载 → 模型训练 → 评估 → 可视化)
python main.py

# 只跑指定模型
python main.py pipeline --models svd,als,itemcf

# 启动交互界面
streamlit run src/app/app.py
```

## CLI 命令

```bash
python main.py                              # 一键端到端 (全部 7 模型)
python main.py pipeline --models svd,als    # 指定模型 + 持久化
python main.py pipeline --save models_out/  # 训练后保存模型

python main.py run --config configs/svd_ml1m.yaml          # 单次实验
python main.py compare configs/svd_ml1m.yaml configs/als_ml1m.yaml  # 逐指标对比
python main.py grid --config configs/svd_grid_ml1m.yaml    # 网格搜索 (粗筛→细化)
```

## 目录结构

```
movie-rec/
├── main.py                     # CLI 入口 (pipeline / run / compare / grid)
├── data_raw/
│   ├── ml-1m/                  # MovieLens 1M 原始数据
│   └── imdb_cache.json         # IMDb 元数据缓存 (89.5% 覆盖)
├── configs/                    # YAML 实验配置 (单次 + 网格搜索)
├── scripts/
│   └── build_imdb_cache.py     # IMDb 缓存离线构建
├── src/
│   ├── data/                   # 模块一: 数据处理
│   │   ├── schema.py           #   数据契约 (Dataset + Split)
│   │   ├── loader.py           #   数据加载 + IMDb 融合 + 划分
│   │   ├── imdb.py             #   IMDb 多源融合 (genres 并集)
│   │   ├── cleaning.py         #   数据清洗 + 质量报告
│   │   └── features.py         #   特征工程 (编码 / 统计量)
│   ├── models/                 # 模块二: 模型训练
│   │   ├── base.py             #   统一接口 + 持久化 (save/load)
│   │   ├── popularity.py       #   热门基线 (贝叶斯平滑)
│   │   ├── itemcf.py           #   物品协同过滤 (top-N 邻居截断)
│   │   ├── svd.py              #   SVD 矩阵分解 (mini-batch SGD)
│   │   ├── als.py              #   ALS 交替最小二乘 (闭式解)
│   │   ├── content.py          #   基于内容 (TF-IDF)
│   │   ├── hybrid.py           #   混合融合 (SVD + Content 加权)
│   │   └── coldstart.py        #   冷启动双通路
│   ├── evaluation/             # 模块三: 评估
│   │   ├── metrics.py          #   三层指标库
│   │   ├── evaluator.py        #   评估器 + 冷启动分层
│   │   └── visualize.py        #   模型对比可视化
│   ├── experiment/             # 实验对比 & 超参调优
│   │   ├── runner.py           #   YAML config → 实验 → metrics
│   │   ├── compare.py          #   逐指标对比 (delta / 判定)
│   │   └── grid.py             #   网格搜索 (粗筛 → 细化)
│   └── app/                    # 模块四: 界面
│       └── app.py              #   Streamlit Dashboard
├── docs/
│   ├── data_preprocess/        # 数据处理文档
│   │   ├── data.md
│   │   ├── data_lineage.md
│   │   └── features.md
│   └── models/                 # 模型文档
│       ├── model_design.md
│       └── model_comparison.md
└── tests/
    └── test_pipeline.py
```

## 数据源

| 数据源 | 来源 | 内容 |
|---|---|---|
| **MovieLens 1M** | GroupLens Research | 6040 用户 × 3883 电影 × 1,000,209 评分 + 用户画像 |
| **IMDb 公开数据集** | datasets.imdbws.com | 物品侧权威元数据 (规范化 genres / runtime) |

两者通过 title+year 连接，genres 取并集，覆盖 **89.5%** 的电影。详见 [docs/data_preprocess/](docs/data_preprocess/)。

## 模型总览

| # | 模型 | 类别 | RMSE↓ | Recall@10↑ | Coverage↑ | 冷用户服务率 |
|---|---|---|---|---|---|---|
| 1 | Popularity | 基线 | 0.990 | 0.020 | 1.5% | 100% |
| 2 | ItemCF | 协同过滤 | 1.008 | **0.082** | 30.8% | 0% |
| 3 | SVD | 协同过滤 (SGD) | **0.890** | 0.028 | 16.8% | 0% |
| 4 | ALS | 协同过滤 (闭式) | 1.210 | 0.001 | 34.4% | 0% |
| 5 | Content | 内容画像 | 1.308 | 0.011 | **79.5%** | 100%* |
| 6 | Hybrid | 混合融合 | 0.922 | 0.031 | 24.7% | 100% |
| 7 | ColdStart | 冷启动 | 1.308 | 0.011 | 79.5% | 100% |

- **SVD**: 评分预测 RMSE 最优，适合"猜你评分"
- **ItemCF**: Top-N 排序最强，适合"猜你想看"
- **ALS**: 交替最小二乘闭式解，SVD 的优化方法对比基线
- **Content**: 覆盖率最高、触达长尾与冷启动
- **Hybrid**: CF 主导 (α=0.75) + 内容补充，综合均衡
- **ColdStart**: 训练集未见用户 100% 服务 (纯 CF 为 0%)

详见 [docs/models/](docs/models/)。

## 设计原则

- **接口优先**: 数据契约 (`schema.py`)、模型接口 (`base.py`) 锁死，四模块解耦
- **多源融合**: MovieLens (行为) + IMDb (内容) 互补，genres 并集量化增益显著
- **三层评估**: 评分预测 (RMSE/MAE) + Top-N 排序 (Recall/NDCG/MAP) + 业务指标 (Coverage/Diversity/Novelty)
- **可复现调优**: YAML 配置驱动，网格搜索粗筛→细化，实验结果可对比、可追溯
