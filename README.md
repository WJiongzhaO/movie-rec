# Movie Recommendation System (MovieLens 1M + IMDb)

> Data Mining and Analytics · Final Project · End-to-End Data Mining Application

A full-stack movie recommendation system covering **Data Processing -> Model Training -> Evaluation -> Interactive Interface**.
Integrates MovieLens 1M user behavior with IMDb item metadata, providing **7 recommendation algorithms** with unified comparison and tuning.

## Quick Start

```bash
pip install -r requirements.txt

# One-click end-to-end pipeline (data -> models -> evaluation -> visualization)
python main.py

# Run specific models only
python main.py pipeline --models svd,als,itemcf

# Launch interactive dashboard
streamlit run src/app/app.py
```

## CLI Reference

```bash
python main.py                              # One-click full pipeline (all 7 models)
python main.py pipeline --models svd,als    # Specific models + persistence
python main.py pipeline --save models_out/  # Save trained models

python main.py run --config configs/svd_ml1m.yaml          # Single experiment
python main.py compare configs/svd_ml1m.yaml configs/als_ml1m.yaml  # Metric-by-metric comparison
python main.py grid --config configs/svd_grid_ml1m.yaml    # Grid search (coarse -> refine)
```

## Project Structure

```
movie-rec/
├── main.py                     # CLI entry (pipeline / run / compare / grid)
├── data_raw/
│   ├── ml-1m/                  # MovieLens 1M raw data
│   └── imdb_cache.json         # IMDb metadata cache (89.5% coverage)
├── configs/                    # YAML experiment configs (single + grid search)
├── scripts/
│   └── build_imdb_cache.py     # IMDb cache offline builder
├── src/
│   ├── data/                   # Module 1: Data Processing
│   │   ├── schema.py           #   Data contract (Dataset + Split)
│   │   ├── loader.py           #   Data loading + IMDb fusion + splits
│   │   ├── imdb.py             #   IMDb multi-source fusion (genres union)
│   │   ├── cleaning.py         #   Data cleaning + quality report
│   │   └── features.py         #   Feature engineering (encoding / statistics)
│   ├── models/                 # Module 2: Model Training
│   │   ├── base.py             #   Unified interface + persistence (save/load)
│   │   ├── popularity.py       #   Popularity baseline (Bayesian smoothing)
│   │   ├── itemcf.py           #   Item-based CF (top-N neighbor truncation)
│   │   ├── svd.py              #   SVD matrix factorization (mini-batch SGD)
│   │   ├── als.py              #   ALS alternating least squares (closed-form)
│   │   ├── content.py          #   Content-based (TF-IDF)
│   │   ├── hybrid.py           #   Hybrid fusion (SVD + Content weighted)
│   │   └── coldstart.py        #   Cold-start dual-pathway
│   ├── evaluation/             # Module 3: Evaluation
│   │   ├── metrics.py          #   3-layer metrics library
│   │   ├── evaluator.py        #   Evaluator + cold-start stratified
│   │   ├── significance.py     #   Statistical significance testing
│   │   ├── error_analysis.py   #   Error stratified analysis
│   │   └── visualize.py        #   Multi-model visualization (7 chart types)
│   ├── experiment/             # Experiment comparison & tuning
│   │   ├── runner.py           #   YAML config -> experiment -> metrics
│   │   ├── compare.py          #   Metric-by-metric comparison (delta / judgment)
│   │   └── grid.py             #   Grid search (coarse -> refine)
│   └── app/                    # Module 4: Interface
│       └── app.py              #   Streamlit Dashboard (4 pages)
├── docs/
│   ├── data_preprocess/        # Data documentation
│   │   ├── data.md
│   │   ├── data_lineage.md
│   │   └── features.md
│   ├── models/                 # Model documentation
│   │   ├── model_design.md
│   │   ├── model_comparison.md
│   │   └── experiment.md
│   └── evaluation/             # Evaluation documentation
│       └── evaluation_design.md
└── tests/
    └── test_pipeline.py
```

## Data Sources

| Source | Provider | Content |
|---|---|---|
| **MovieLens 1M** | GroupLens Research | 6,040 users x 3,883 movies x 1,000,209 ratings + user profiles |
| **IMDb Public Dataset** | datasets.imdbws.com | Item-side authoritative metadata (normalized genres / runtime) |

Data sources are joined by title+year, genres are merged via set union, covering **89.5%** of movies.
See [docs/data_preprocess/](docs/data_preprocess/) for full documentation.

## Models at a Glance

| # | Model | Category | RMSE | Recall@10 | Coverage | Cold-Start Serve Rate |
|---|---|---|---|---|---|---|
| 1 | Popularity | Baseline | 0.990 | 0.020 | 1.5% | 100% |
| 2 | ItemCF | Collaborative Filtering | 1.008 | **0.082** | 30.8% | 0% |
| 3 | SVD | Collaborative Filtering (SGD) | **0.890** | 0.028 | 16.8% | 0% |
| 4 | ALS | Collaborative Filtering (Closed-form) | 1.210 | 0.001 | 34.4% | 0% |
| 5 | Content | Content-Based | 1.308 | 0.011 | **79.5%** | 100%* |
| 6 | Hybrid | Hybrid Fusion | 0.922 | 0.031 | 24.7% | 100% |
| 7 | ColdStart | Cold-Start | 1.308 | 0.011 | 79.5% | 100% |

- **SVD**: Best rating prediction (RMSE 0.890), suitable for "guess my rating"
- **ItemCF**: Best Top-N ranking (Recall@10 0.082, NDCG@10 0.115), suitable for "guess what I like"
- **ALS**: Closed-form alternating least squares, optimization-method comparison baseline
- **Content**: Highest coverage (79.5%), reaches long-tail and cold-start items
- **Hybrid**: CF-dominant (alpha=0.75) + content supplement, balanced accuracy and diversity
- **ColdStart**: 100% serve rate for unseen users (pure CF = 0%)

See [docs/models/](docs/models/) for detailed analysis.

## Design Principles

- **Interface First**: Data contract (`schema.py`) and model interface (`base.py`) are locked first, enabling parallel module development
- **Multi-Source Fusion**: MovieLens (behavior) + IMDb (content) complement each other; genres union provides measurable gains
- **Three-Layer Evaluation**: Rating prediction (RMSE/MAE) + Top-N ranking (Recall/NDCG/MAP) + Business metrics (Coverage/Diversity/Novelty)
- **Reproducible Tuning**: YAML-driven configs, grid search coarse-to-refine, experiment results comparable and traceable
