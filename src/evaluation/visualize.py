"""
Result visualization for reports and defense presentations.
============================================================
Chart types:
  - Multi-model metric comparison (grouped bar chart, 3 subplots)
  - Cold-start service capability comparison
  - Multi-K ranking quality curves (Recall/NDCG/HitRate @5,10,15,20)
  - Rating error distribution (histogram + by-rating bar)
  - User activity / item popularity stratified analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
           "Source Han Sans CN", "SimHei"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "experiments" / "figures"

MODEL_ORDER = ["svd", "als", "itemcf", "hybrid", "content", "coldstart", "popularity"]
MODEL_LABEL = {
    "svd": "SVD", "als": "ALS", "itemcf": "ItemCF", "hybrid": "Hybrid",
    "content": "Content", "coldstart": "ColdStart", "popularity": "Popularity",
}


# ===================================================================
# 1. Multi-model comparison
# ===================================================================

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    out_dir: str | Path | None = None,
) -> Path:
    """Grouped bar chart: 3 subplots for rating / ranking / business metrics."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    order = [m for m in MODEL_ORDER if m in results]
    labels = [MODEL_LABEL.get(m, m) for m in order]

    groups = [
        ("Rating prediction (lower better)", ["rmse", "mae"]),
        ("Top-N ranking (higher better)", ["recall@10", "ndcg@10", "map@10"]),
        ("Business metrics (higher better)", ["coverage", "diversity", "novelty"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (title, metrics) in zip(axes, groups):
        available = [m for m in metrics if m in results.get(order[0], {})]
        if not available:
            continue
        x = range(len(labels))
        width = 0.8 / max(len(available), 1)
        for i, met in enumerate(available):
            vals = [results[m].get(met, 0) for m in order]
            ax.bar([p + i * width for p in x], vals, width, label=met)
        ax.set_title(title)
        ax.set_xticks([p + width * (len(available) - 1) / 2 for p in x])
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Multi-model comparison (MovieLens 1M + IMDb)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def plot_coldstart_comparison(
    results: Dict[str, Dict[str, float]],
    out_dir: str | Path | None = None,
) -> Path | None:
    """Cold-start: serve rate + hit rate per model."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cold_metrics = ["cold_serve_rate", "cold_hitrate@10"]
    order = [m for m in MODEL_ORDER if m in results]
    available = [m for m in cold_metrics if m in results.get(order[0], {})]
    if not available:
        return None

    labels = [MODEL_LABEL.get(m, m) for m in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels)); width = 0.35
    for i, met in enumerate(available):
        vals = [results[m].get(met, 0) for m in order]
        ax.bar([p + i * width for p in x], vals, width, label=met)
    ax.set_title("Cold-start: serve rate for unseen users")
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "coldstart_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


# ===================================================================
# 2. Multi-K ranking curves
# ===================================================================

def plot_multik_curves(
    model,
    split,
    dataset,
    ks: List[int] = None,
    out_dir: str | Path | None = None,
    model_label: str = "",
):
    """Recall/NDCG/HitRate vs K for a single model."""
    from . import metrics as M

    ks = ks or [5, 10, 15, 20]
    test = split.test
    relevant_by_user: Dict[int, set] = {}
    for u, g in test.groupby("user_id"):
        rel = set(g[g["rating"] >= 4.0]["item_id"])
        if rel:
            relevant_by_user[u] = rel

    data = {"k": ks, "recall": [], "ndcg": [], "hitrate": []}
    for k in ks:
        rv, nv, hv = [], [], []
        for u, rel in relevant_by_user.items():
            recs = model.recommend(int(u), k=k)
            if not recs:
                continue
            rv.append(M.recall_at_k(recs, rel, k))
            nv.append(M.ndcg_at_k(recs, rel, k))
            hv.append(M.hitrate_at_k(recs, rel, k))
        data["recall"].append(round(float(np.mean(rv)), 5) if rv else 0.0)
        data["ndcg"].append(round(float(np.mean(nv)), 5) if nv else 0.0)
        data["hitrate"].append(round(float(np.mean(hv)), 5) if hv else 0.0)

    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, title) in zip(axes, [("recall", "Recall@K"), ("ndcg", "NDCG@K"), ("hitrate", "HitRate@K")]):
        ax.plot(data["k"], data[key], "o-", color="#6366f1", markersize=6)
        ax.set_title(title); ax.set_xlabel("K"); ax.set_xticks(data["k"]); ax.grid(alpha=0.3)

    prefix = f" — {model_label}" if model_label else ""
    fig.suptitle(f"Ranking quality vs K{prefix}", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"multik_{model_label or 'model'}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def plot_multik_models(
    model_results: Dict[str, dict],
    out_dir: str | Path | None = None,
):
    """Overlay multi-K curves for multiple models."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, [("recall", "Recall@K"), ("ndcg", "NDCG@K"), ("hitrate", "HitRate@K")]):
        k_ref = None
        for m_name, data in model_results.items():
            if key in data and data[key]:
                ax.plot(data["k"], data[key], "o-", label=m_name, markersize=5)
                k_ref = data["k"]
        ax.set_title(title); ax.set_xlabel("K")
        if k_ref: ax.set_xticks(k_ref)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Multi-K ranking curves", fontsize=13)
    fig.tight_layout()
    path = out_dir / "multik_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


# ===================================================================
# 3. Rating error distribution
# ===================================================================

def plot_error_distribution(
    model,
    split,
    out_dir: str | Path | None = None,
    model_label: str = "",
):
    """Histogram of rating errors + mean error by true rating."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    test = split.test
    y_true = test["rating"].to_numpy(dtype=np.float32)
    y_pred = np.array([
        model.predict(int(u), int(i))
        for u, i in test[["user_id", "item_id"]].itertuples(index=False)
    ])
    errors = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(errors, bins=50, color="#6366f1", alpha=0.8, edgecolor="white")
    ax1.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="zero")
    over_pct = 100 * np.mean(y_pred > y_true)
    under_pct = 100 * np.mean(y_pred < y_true)
    ax1.set_title(f"Error histogram (over-est {over_pct:.1f}% / under-est {under_pct:.1f}%)")
    ax1.set_xlabel("True - Predicted"); ax1.set_ylabel("Count"); ax1.legend()

    rating_vals = sorted(test["rating"].unique())
    mean_errs = []
    for r in rating_vals:
        m = test["rating"] == r
        mean_errs.append(float(errors[m].mean()) if m.sum() > 0 else 0.0)
    colors = ["#ef4444" if e > 0 else "#22c55e" for e in mean_errs]
    ax2.bar(rating_vals, mean_errs, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Mean error by true rating"); ax2.set_xlabel("True rating")

    prefix = f" — {model_label}" if model_label else ""
    fig.suptitle(f"Rating error analysis{prefix}", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"error_dist_{model_label or 'model'}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


# ===================================================================
# 4. Stratified analysis (user activity / item popularity)
# ===================================================================

def plot_activity_breakdown(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
    title: str = "",
):
    """Dual-axis bar chart: RMSE + Recall per user activity bucket."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = range(len(df)); w = 0.35

    ax1.bar([p - w/2 for p in x], df["rmse"], w, color="#6366f1", label="RMSE")
    ax1.set_ylabel("RMSE", color="#6366f1")
    ax1.tick_params(axis="y", labelcolor="#6366f1")

    ax2 = ax1.twinx()
    rc = [c for c in df.columns if "recall" in c][0]
    ax2.bar([p + w/2 for p in x], df[rc], w, color="#f59e0b", label="Recall")
    ax2.set_ylabel("Recall", color="#f59e0b")
    ax2.tick_params(axis="y", labelcolor="#f59e0b")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(
        [f"{r['bucket']}\n({r['mean_train_ratings']:.0f} ratings)" for _, r in df.iterrows()],
        fontsize=8)
    ax1.set_title(f"User activity stratified analysis {title}".strip())
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "user_activity_breakdown.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def plot_popularity_breakdown(
    df: pd.DataFrame,
    out_dir: str | Path | None = None,
):
    """Bar chart: hit rate per item popularity bucket."""
    out_dir = Path(out_dir) if out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    colors = ["#ef4444", "#f59e0b", "#22c55e", "#6366f1"][:len(df)]
    ax.bar(x, df["hit_rate"], color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{r['bucket']}\n({r['mean_ratings']:.0f} ratings)" for _, r in df.iterrows()],
        fontsize=9)
    ax.set_ylabel("HitRate@10")
    ax.set_title("Item popularity stratified: cold vs popular items")
    ax.grid(axis="y", alpha=0.3)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["hit_rate"] + 0.01, f'{r["hit_rate"]:.3f}', ha="center", fontsize=9)

    fig.tight_layout()
    path = out_dir / "item_popularity_breakdown.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    from src.data.loader import load_data, train_test_split_by_time
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.error_analysis import by_user_activity, by_item_popularity
    from src.models.svd import SVDRecommender
    from src.models.itemcf import ItemCFRecommender
    from src.models.hybrid import HybridRecommender
    from src.models.content import ContentRecommender
    from src.models.coldstart import ColdStartRecommender
    from src.models.popularity import PopularityRecommender
    from src.models.als import ALSRecommender

    print("=" * 50)
    print("Generating all evaluation charts ...")
    print("=" * 50)

    ds = load_data()
    split = train_test_split_by_time(ds, test_ratio=0.2)
    ev = Evaluator(k=10)

    MODELS = {
        "popularity": (PopularityRecommender, {}),
        "itemcf":     (ItemCFRecommender, {"topk_neighbors": 120}),
        "svd":        (SVDRecommender, {"n_factors": 50, "n_epochs": 20, "lr": 0.005, "reg": 0.02, "seed": 42}),
        "als":        (ALSRecommender, {"n_factors": 50, "reg": 0.1, "n_epochs": 10}),
        "content":    (ContentRecommender, {}),
        "hybrid":     (HybridRecommender, {"alpha": 0.75, "cf_params": {"n_factors": 50, "n_epochs": 20, "lr": 0.005, "reg": 0.02, "seed": 42}}),
        "coldstart":  (ColdStartRecommender, {}),
    }

    all_results = {}
    multik_data = {}

    for name, (cls, kwargs) in MODELS.items():
        print(f"\n[{name}] training ...")
        m = cls(**kwargs).fit(split, ds)
        metrics = ev.evaluate(m, split, ds)
        all_results[name] = metrics
        print(f"  RMSE={metrics.get('rmse', '—'):.4f}  Recall@10={metrics.get('recall@10', '—'):.4f}")

        # Multi-K curves
        print(f"  multi-K ...")
        plot_multik_curves(m, split, ds, model_label=name)

        # Error dist (key models only)
        if name in ("svd", "itemcf", "hybrid"):
            print(f"  error dist ...")
            plot_error_distribution(m, split, model_label=name)

        # Stratified (SVD only)
        if name == "svd":
            print(f"  activity breakdown ...")
            ua = by_user_activity(m, split, ds, n_buckets=4)
            plot_activity_breakdown(ua, title="SVD")
            print(f"  popularity breakdown ...")
            ip = by_item_popularity(m, split, ds, n_buckets=4)
            plot_popularity_breakdown(ip)

    # Aggregate charts
    print(f"\n[aggregate] model comparison ...")
    plot_model_comparison(all_results)

    print(f"\n{'='*50}")
    print(f"Done. Charts saved to: {FIGURES_DIR}")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        sz = f.stat().st_size
        print(f"  {f.name}  ({sz:,} bytes)")
