"""
实验执行器 (轻量版)
====================
读 YAML config → 加载数据 → 训练模型 → 评估 → 返回 metrics dict。
无持久化注册表, 纯函数式。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import yaml

from ..data.loader import load_data, train_test_split_by_time, train_test_split_with_cold_users
from ..evaluation.evaluator import Evaluator
from ..models.als import ALSRecommender
from ..models.coldstart import ColdStartRecommender
from ..models.content import ContentRecommender
from ..models.hybrid import HybridRecommender
from ..models.itemcf import ItemCFRecommender
from ..models.popularity import PopularityRecommender
from ..models.svd import SVDRecommender

_MODEL_CLASSES = {
    "popularity": PopularityRecommender,
    "itemcf": ItemCFRecommender,
    "svd": SVDRecommender,
    "als": ALSRecommender,
    "content": ContentRecommender,
    "hybrid": HybridRecommender,
    "coldstart": ColdStartRecommender,
}


def run_from_config(config: Union[dict, str, Path]) -> dict:
    """读配置, 跑一次实验, 返回 {model, params, metrics, elapsed_s, note}。

    配置结构 (YAML):
      data:   {source: movielens}    # source 固定为 movielens
      split:  {test_ratio: 0.2, cold_user_ratio: null}
      model:  {name: svd, params: {n_factors: 50, ...}}
      eval:   {k: 10, relevance_threshold: 4.0}
      note:   "实验说明"
    """
    if isinstance(config, (str, Path)):
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    else:
        cfg = config

    # 1. 数据 (固定真实数据)
    dataset = load_data()

    # 2. 划分
    split_cfg = dict(cfg.get("split", {}))
    cold_user_ratio = split_cfg.pop("cold_user_ratio", None)
    test_ratio = split_cfg.pop("test_ratio", 0.2)
    if cold_user_ratio is not None:
        split = train_test_split_with_cold_users(
            dataset, test_ratio=test_ratio, cold_user_ratio=cold_user_ratio)
    else:
        split = train_test_split_by_time(dataset, test_ratio=test_ratio)

    # 3. 模型
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    model_cls = _MODEL_CLASSES[model_name]
    model = model_cls(**model_cfg.get("params", {}))

    t0 = time.time()
    model.fit(split, dataset)
    train_s = time.time() - t0

    # 4. 评估
    eval_cfg = cfg.get("eval", {})
    evaluator = Evaluator(**eval_cfg)
    metrics = evaluator.evaluate(model, split, dataset)
    if split.cold_user_test is not None:
        metrics.update(evaluator.evaluate_cold_users(model, split, dataset))

    return {
        "model": model_name,
        "params": model_cfg.get("params", {}),
        "metrics": metrics,
        "train_s": round(train_s, 1),
        "note": cfg.get("note", ""),
    }
