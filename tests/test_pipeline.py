"""
端到端自动化测试
================
验证完整数据挖掘管道闭环:数据加载 → 模型训练 → 评估,仅使用真实数据(MovieLens 1M + IMDb)。

运行: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import (
    load_data,
    train_test_split_by_time,
    train_test_split_with_cold_users,
)
from src.data.schema import Dataset
from src.evaluation.evaluator import Evaluator
from src.models.als import ALSRecommender
from src.models.content import ContentRecommender
from src.models.coldstart import ColdStartRecommender
from src.models.hybrid import HybridRecommender
from src.models.itemcf import ItemCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender

ALL_MODELS = {
    "popularity": PopularityRecommender,
    "itemcf": ItemCFRecommender,
    "svd": SVDRecommender,
    "als": ALSRecommender,
    "content": ContentRecommender,
    "hybrid": HybridRecommender,
    "coldstart": ColdStartRecommender,
}

# 快速训练参数 (测试用, 降低 epoch 数)
_FAST_KWARGS = {
    "svd": {"n_factors": 10, "n_epochs": 2},
    "als": {"n_factors": 10, "n_epochs": 3},
    "hybrid": {"cf_params": {"n_factors": 10, "n_epochs": 2}},
}


def test_load_data():
    """数据加载产出符合契约的 Dataset。"""
    ds = load_data()
    assert isinstance(ds, Dataset)
    s = ds.summary()
    assert s["n_users"] > 0 and s["n_items"] > 0 and s["n_ratings"] > 0
    # IMDb 融合应已生效
    assert "imdb_genres" in ds.items.columns or "runtime" in ds.items.columns


def test_split():
    """按时间留出划分:训练+测试 = 全部。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    assert len(split.train) > 0 and len(split.test) > 0
    assert len(split.train) + len(split.test) == len(ds.ratings)


def test_models_train_and_recommend():
    """全部 7 个模型可训练并产出推荐结果。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    for name, cls in ALL_MODELS.items():
        kwargs = _FAST_KWARGS.get(name, {})
        m = cls(**kwargs).fit(split, ds)
        recs = m.recommend(int(ds.ratings.iloc[0]["user_id"]), k=5)
        assert isinstance(recs, list)
        pred = m.predict(
            int(ds.ratings.iloc[0]["user_id"]),
            int(ds.ratings.iloc[0]["item_id"]),
        )
        assert isinstance(pred, float)


def test_evaluator_returns_multi_layer_metrics():
    """三层指标 + 可靠性诊断字段全部产出。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    m = ItemCFRecommender(topk_neighbors=30).fit(split, ds)
    metrics = Evaluator(k=10).evaluate(m, split, ds)
    # L1
    assert "rmse" in metrics and "mae" in metrics
    # L2
    assert any("recall@" in k for k in metrics)
    # L3
    assert "coverage" in metrics and "diversity" in metrics
    # 可靠性诊断
    assert "eval_users" in metrics
    assert "eval_user_coverage" in metrics
    assert "eval_rel_size_mean" in metrics


def test_coldstart_split_and_stratified_eval():
    """冷启动分层评估:纯 CF 无法服务冷用户,coldstart 能兜底。"""
    ds = load_data()
    split = train_test_split_with_cold_users(
        ds, test_ratio=0.2, cold_user_ratio=0.1, seed=11
    )
    assert split.cold_user_test is not None and len(split.cold_user_test) > 0
    cold_users = set(split.cold_user_test["user_id"])
    assert cold_users.isdisjoint(set(split.train["user_id"]))

    ev = Evaluator(k=10)
    svd = SVDRecommender(n_factors=10, n_epochs=2).fit(split, ds)
    cold = ColdStartRecommender().fit(split, ds)
    svd_cold = ev.evaluate_cold_users(svd, split, ds)
    cs_cold = ev.evaluate_cold_users(cold, split, ds)
    assert cs_cold["cold_serve_rate"] > svd_cold["cold_serve_rate"]


def test_coldstart_preference_questionnaire():
    """新用户偏好问卷 → 首屏推荐。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    m = ColdStartRecommender().fit(split, ds)
    recs = m.recommend_for_preferences(["Action", "Comedy"], k=8)
    assert isinstance(recs, list) and len(recs) <= 8


def test_imdb_multisource_fusion():
    """IMDb 第二数据源融合:genres 并集、旁路列正确写入,且不修改入参。"""
    import json
    import tempfile

    import pandas as pd

    from src.data.imdb import cache_status, enrich_items_with_imdb
    from src.data.schema import ITEM_COL

    items = pd.DataFrame(
        {
            ITEM_COL: [1, 2, 3],
            "title": ["A (1999)", "B (2001)", "C (2010)"],
            "genres": ["Action|Comedy", "Drama", ""],
        }
    )
    cache = {
        "1": {"tconst": "tt001", "imdb_genres": "Action,Thriller", "runtime": "120"},
        "2": {"tconst": "tt002", "imdb_genres": "Romance", "runtime": "95"},
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cache, f)
        cache_path = f.name

    out = enrich_items_with_imdb(items, cache_path=cache_path)
    assert "imdb_genres" not in items.columns and "runtime" not in items.columns
    assert "imdb_genres" in out.columns and "runtime" in out.columns

    by_id = {int(r[ITEM_COL]): r for _, r in out.iterrows()}
    g1 = set(by_id[1]["genres"].split("|"))
    assert g1 == {"Action", "Comedy", "Thriller"}
    assert int(by_id[1]["runtime"]) == 120

    g2 = set(by_id[2]["genres"].split("|"))
    assert g2 == {"Drama", "Romance"}
    assert int(by_id[2]["runtime"]) == 95

    assert by_id[3]["imdb_genres"] == ""
    assert pd.isna(by_id[3]["runtime"])

    st = cache_status(cache_path=cache_path)
    assert st["imdb_matched_items"] == 2


def test_hybrid_fusion_runnable():
    """Hybrid 模型:两侧子模型均训练、融合推荐可产出。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    m = HybridRecommender(
        alpha=0.75,
        cf_params={"n_factors": 10, "n_epochs": 2},
    ).fit(split, ds)
    recs = m.recommend(int(ds.ratings.iloc[0]["user_id"]), k=5)
    assert len(recs) == 5
    metrics = Evaluator(k=10).evaluate(m, split, ds)
    assert "rmse" in metrics and "recall@10" in metrics


def test_als_comparable_to_svd():
    """ALS 模型: 可训练, 指标在合理范围 (RMSE < 2.0)。"""
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    m = ALSRecommender(n_factors=10, n_epochs=3).fit(split, ds)
    metrics = Evaluator(k=10).evaluate(m, split, ds)
    assert "rmse" in metrics
    assert metrics["rmse"] < 2.0
    recs = m.recommend(int(ds.ratings.iloc[0]["user_id"]), k=5)
    assert len(recs) == 5


def test_model_persistence_roundtrip():
    """所有模型保存→加载后 predict 输出一致。"""
    import tempfile, os
    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    td = tempfile.mkdtemp()

    for name, cls in ALL_MODELS.items():
        kwargs = _FAST_KWARGS.get(name, {})
        m = cls(**kwargs).fit(split, ds)
        path = os.path.join(td, f"{name}.model")
        m.save(path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        m2 = type(m).load(path)
        # 抽样验证预测一致性
        for uid in [1, 100, 500]:
            for iid in [1, 50, 200]:
                assert abs(m.predict(uid, iid) - m2.predict(uid, iid)) < 1e-5, \
                    f"{name}: predict mismatch for ({uid},{iid})"

    import shutil; shutil.rmtree(td)


if __name__ == "__main__":
    test_load_data()
    test_split()
    test_models_train_and_recommend()
    test_evaluator_returns_multi_layer_metrics()
    test_coldstart_split_and_stratified_eval()
    test_coldstart_preference_questionnaire()
    test_imdb_multisource_fusion()
    test_hybrid_fusion_runnable()
    test_als_comparable_to_svd()
    test_model_persistence_roundtrip()
    print("\n[ALL TESTS PASSED]")
