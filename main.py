#!/usr/bin/env python3
"""
电影推荐系统 —— 一键端到端管道
================================
自动完成: 数据加载 → 模型训练 → 三层评估 → 结果汇总 → 可视化图表。

数据源: MovieLens 1M (GroupLens) + IMDb 元数据融合
模型: Popularity / ItemCF / SVD / Content / Hybrid / ColdStart

用法:
    python main.py                     # 完整管道(全部模型)
    python main.py --models svd,itemcf  # 只跑指定模型
    python main.py --no-viz             # 跳过可视化
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import (
    load_data,
    train_test_split_by_time,
    train_test_split_with_cold_users,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualize import plot_model_comparison, plot_coldstart_comparison
from src.models.als import ALSRecommender
from src.models.base import BaseRecommender
from src.models.content import ContentRecommender
from src.models.coldstart import ColdStartRecommender
from src.models.hybrid import HybridRecommender
from src.models.itemcf import ItemCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender

# 全部可用模型(名称 → 类)
ALL_MODELS: dict[str, type[BaseRecommender]] = {
    "popularity": PopularityRecommender,
    "itemcf": ItemCFRecommender,
    "svd": SVDRecommender,
    "als": ALSRecommender,
    "content": ContentRecommender,
    "hybrid": HybridRecommender,
    "coldstart": ColdStartRecommender,
}

# 模型一句话定位
MODEL_BLURB = {
    "popularity": "热门基线 · 非个性化下限参照",
    "itemcf": "物品协同过滤 · Top-N 排序最强",
    "svd": "矩阵分解(SGD) · 评分预测 RMSE 最优",
    "als": "交替最小二乘 · 闭式解收敛, 无需调学习率",
    "content": "基于内容 · 覆盖率最高、缓解冷启动",
    "hybrid": "混合融合(SVD+Content) · 综合均衡",
    "coldstart": "冷启动通路 · 新用户也能服务",
}

# SVD 最优参数(来自完整训练复核)
SVD_PARAMS = {"n_factors": 50, "n_epochs": 20, "lr": 0.005, "reg": 0.02, "seed": 42}
# ALS 参数 (与 SVD 对齐 n_factors/reg 以公平对比)
ALS_PARAMS = {"n_factors": 50, "reg": 0.1, "n_epochs": 15, "seed": 42}
# Hybrid 最优参数(alpha=0.75 网格调优)
HYBRID_PARAMS = {
    "alpha": 0.75,
    "cf_params": SVD_PARAMS,
    "content_params": {"like_threshold": 4.0},
}
# ItemCF 最优参数(topk_neighbors=120)
ITEMCF_PARAMS = {"topk_neighbors": 120}


def build_model(name: str) -> BaseRecommender:
    """根据名称构建模型(使用调优后的参数)。"""
    if name == "svd":
        return SVDRecommender(**SVD_PARAMS)
    if name == "als":
        return ALSRecommender(**ALS_PARAMS)
    if name == "hybrid":
        return HybridRecommender(**HYBRID_PARAMS)
    if name == "itemcf":
        return ItemCFRecommender(**ITEMCF_PARAMS)
    return ALL_MODELS[name]()


def run_pipeline(
    model_names: list[str] | None = None,
    do_viz: bool = True,
    cold_user_ratio: float = 0.1,
    save_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """执行完整管道,返回 {model_name: {metric: value, ...}}。

    步骤:
      1. 加载 MovieLens 1M + IMDb 融合数据
      2. 按时间留出划分(含冷用户子集)
      3. 逐模型训练 + 三层评估
      4. 打印汇总对比表
      5. 生成可视化图表
    """
    names = model_names or list(ALL_MODELS)
    evaluator = Evaluator(k=10, relevance_threshold=4.0)

    # ---- 1. 数据加载 ----
    print("=" * 60)
    print("电影推荐系统 · 端到端管道")
    print("=" * 60)
    print("\n[1/4] 加载数据 (MovieLens 1M + IMDb) ...")
    t0 = time.time()
    dataset = load_data()
    print(f"  用户={dataset.n_users}, 物品={dataset.n_items}, "
          f"评分={dataset.n_ratings}, 密度={dataset.density:.4f}")
    meta = dataset.meta
    if meta.get("imdb_used"):
        print(f"  IMDb 融合: {meta.get('imdb_matched_items')} 部电影"
              f" ({meta['imdb_matched_items'] / dataset.n_items:.1%})")

    # 质量报告摘要
    q = meta.get("quality", {})
    if q:
        dq = q.get("dedup", {})
        lq = q.get("low_quality_flag", {})
        mv = q.get("missing_values", {})
        imv = mv.get("items", {})
        print(f"  数据清洗: 重复评分={dq.get('dedup_removed',0)}条, "
              f"runtime缺失={imv.get('runtime','—')}条, "
              f"冷门物品={lq.get('n_low_popularity_items','—')}个")

    # ---- 2. 数据划分 + 特征工程 ----
    print("\n[2/4] 划分训练/测试集 + 特征工程 ...")
    from src.data.features import build_all_features
    features = build_all_features(dataset)
    print(f"  类型矩阵: {features['genre_matrix'].shape} "
          f"({len(features['genre_names'])} 种类型)")
    print(f"  用户特征: {features['user_features'].shape}")
    print(f"  物品统计: {features['item_stats'].shape}")
    split = train_test_split_with_cold_users(
        dataset, test_ratio=0.2, cold_user_ratio=cold_user_ratio, seed=42
    )
    n_cold = len(split.cold_user_test["user_id"].unique()) if split.cold_user_test is not None else 0
    print(f"  训练集={len(split.train)} 条, 测试集={len(split.test)} 条"
          f", 冷用户={n_cold} 人")

    # ---- 3. 训练 + 评估 ----
    print(f"\n[3/4] 训练 {len(names)} 个模型并评估 ...")
    all_results: dict[str, dict[str, float]] = {}

    for i, name in enumerate(names, 1):
        print(f"  [{i}/{len(names)}] {name} ({MODEL_BLURB.get(name, '')}) ...", end=" ", flush=True)
        t_m = time.time()
        model = build_model(name)
        model.fit(split, dataset)
        # 持久化保存
        if save_dir:
            p = Path(save_dir); p.mkdir(parents=True, exist_ok=True)
            model.save(str(p / f"{name}.model"))
        metrics = evaluator.evaluate(model, split, dataset)
        if split.cold_user_test is not None:
            metrics.update(evaluator.evaluate_cold_users(model, split, dataset))
        all_results[name] = metrics
        elapsed = time.time() - t_m
        rmse = metrics.get("rmse", float("nan"))
        recall = metrics.get("recall@10", float("nan"))
        print(f"RMSE={rmse:.4f}  Recall@10={recall:.4f}  ({elapsed:.1f}s)")

    # ---- 4. 汇总对比 ----
    print("\n[4/4] 结果汇总")
    _print_summary_table(all_results)
    print(f"\n总耗时: {time.time() - t0:.1f}s")

    # ---- 5. 可视化 ----
    if do_viz:
        print("\n生成可视化图表 ...")
        plot_model_comparison(all_results)
        if any("cold_serve_rate" in r for r in all_results.values()):
            plot_coldstart_comparison(all_results)

    print("\n" + "=" * 60)
    print("管道完成。启动交互界面: streamlit run src/app/app.py")
    print("=" * 60)

    return all_results


def _print_summary_table(results: dict[str, dict[str, float]]) -> None:
    """打印模型 × 指标汇总对比表。"""
    key_metrics = [
        ("rmse", "↓"), ("mae", "↓"),
        ("precision@10", "↑"), ("recall@10", "↑"),
        ("ndcg@10", "↑"), ("map@10", "↑"),
        ("coverage", "↑"), ("diversity", "↑"), ("novelty", "↑"),
    ]
    # 标题行
    header = f"{'模型':<14}"
    for m, d in key_metrics:
        header += f" {m:>14}"
    print("\n" + header)
    print("-" * len(header))

    order = ["popularity", "itemcf", "svd", "als", "content", "hybrid", "coldstart"]
    for name in order:
        if name not in results:
            continue
        row = f"{name:<14}"
        for m, _ in key_metrics:
            v = results[name].get(m)
            row += f" {v:>14.4f}" if isinstance(v, float) else f" {'—':>14}"
        print(row)

    # 冷启动对比
    cold_keys = [k for k in results.get("coldstart", {}) if k.startswith("cold_")]
    if cold_keys:
        print(f"\n{'冷启动分层对比':—^40}")
        cold_header = f"{'模型':<14} {'cold_serve_rate':>16} {'cold_hitrate@10':>17}"
        print(cold_header)
        print("-" * len(cold_header))
        for name in order:
            if name not in results:
                continue
            sr = results[name].get("cold_serve_rate", 0)
            hr = results[name].get("cold_hitrate@10", 0)
            print(f"{name:<14} {sr:>16.4f} {hr:>17.4f}")


def main():
    parser = argparse.ArgumentParser(description="电影推荐系统 · CLI")
    sub = parser.add_subparsers(dest="cmd")

    # ── pipeline: 一键端到端 ──
    p_pipe = sub.add_parser("pipeline", help="一键端到端管道 (默认)")
    p_pipe.add_argument("--models", default=None,
                        help="只跑指定模型,逗号分隔 (默认全部)")
    p_pipe.add_argument("--no-viz", action="store_true", help="跳过可视化")
    p_pipe.add_argument("--save", default=None, help="模型保存目录")
    p_pipe.add_argument("--cold-ratio", type=float, default=0.1)

    # ── run: 单次实验 ──
    p_run = sub.add_parser("run", help="从 YAML config 跑一次实验")
    p_run.add_argument("--config", required=True, help="YAML 配置路径")

    # ── compare: 对比两个实验 ──
    p_cmp = sub.add_parser("compare", help="对比两个 YAML config 的实验结果")
    p_cmp.add_argument("config_a", help="基线 YAML")
    p_cmp.add_argument("config_b", help="对比 YAML")

    # ── grid: 网格搜索 ──
    p_grid = sub.add_parser("grid", help="超参网格搜索 (粗筛→细化)")
    p_grid.add_argument("--config", required=True, help="网格搜索 YAML")

    args = parser.parse_args()

    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "compare":
        _cmd_compare(args)
    elif args.cmd == "grid":
        _cmd_grid(args)
    else:
        # 默认: pipeline (兼容无子命令调用)
        model_names = None
        models_arg = getattr(args, 'models', None)
        if models_arg:
            model_names = [n.strip() for n in models_arg.split(",")]
            unknown = set(model_names) - set(ALL_MODELS)
            if unknown:
                print(f"未知模型: {unknown}. 可用: {list(ALL_MODELS)}")
                sys.exit(1)
        run_pipeline(
            model_names=model_names,
            do_viz=not getattr(args, 'no_viz', False),
            cold_user_ratio=getattr(args, 'cold_ratio', 0.1),
            save_dir=getattr(args, 'save', None),
        )


def _cmd_run(args):
    from src.experiment.runner import run_from_config
    result = run_from_config(args.config)
    print(f"\n[模型] {result['model']}  [训练耗时] {result['train_s']}s")
    if result.get("note"):
        print(f"[备注] {result['note']}")
    print(f"[超参] {result['params']}")
    print("[指标]")
    for k, v in sorted(result["metrics"].items()):
        print(f"  {k:<20} = {v}")


def _cmd_compare(args):
    from src.experiment.runner import run_from_config
    from src.experiment.compare import compare

    print(f"[基线] {args.config_a}")
    r_a = run_from_config(args.config_a)
    print(f"[对比] {args.config_b}")
    r_b = run_from_config(args.config_b)

    df = compare(r_a, r_b)
    # 只打印关键指标 (排除诊断字段)
    show = df[~df["metric"].str.startswith("eval_")]
    print(f"\n{'逐指标对比':—^60}")
    print(f"基线: {r_a['model']} | 对比: {r_b['model']}")
    print(show.to_string(index=False))


def _cmd_grid(args):
    from src.experiment.grid import grid_search_from_config
    result = grid_search_from_config(args.config)

    print(f"\n{'='*40}")
    print("粗筛最优超参:", result["coarse_best"])
    if result["refine"] is not None:
        print("细化最优超参:", result["refine_best"])
        print("\n[细化结果]")
        print(result["refine"].to_string(index=False))


if __name__ == "__main__":
    main()
