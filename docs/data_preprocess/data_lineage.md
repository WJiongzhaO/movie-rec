# 数据血缘图

> 可直接复制 Mermaid 代码到 [mermaid.live](https://mermaid.live) 渲染为矢量图，
> 或在 VS Code / GitHub / Typora 中直接预览。

---

## 完整数据血缘 (纵向流程)

```mermaid
flowchart TB
    %% ═══ 数据源 ═══
    subgraph SRC["📦 数据源"]
        ML[("MovieLens 1M<br/>GroupLens Research<br/>6040用户 × 3883电影")]
        IMDB[("IMDb 公开数据集<br/>datasets.imdbws.com<br/>title.basics.tsv.gz")]
    end

    %% ═══ 原始文件 ═══
    subgraph RAW["📄 原始文件"]
        RAT["ratings.dat<br/>1,000,209 行<br/>UserID::MovieID::Rating::Timestamp"]
        MOV["movies.dat<br/>3,883 行<br/>MovieID::Title::Genres"]
        USR["users.dat<br/>6,040 行<br/>UserID::Gender::Age::Occupation::Zip-code"]
    end

    ML --> RAT
    ML --> MOV
    ML --> USR

    %% ═══ IMDb ETL ═══
    subgraph ETL["🔧 IMDb 离线 ETL  scripts/build_imdb_cache.py"]
        E1["1. 过滤 titleType ∈ {movie, tvMovie}"]
        E2["2. norm_title(): 去年份括号 / The,A,An 还原 / 小写去标点"]
        E3["3. 建索引 (norm_title, year) → 记录"]
        E4["4. title+year JOIN movies.dat → 命中 3476/3883 (89.5%)"]
        E5["5. 输出 imdb_cache.json"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    IMDB --> E1
    CACHE[("imdb_cache.json<br/>3476 条")]
    E5 --> CACHE

    %% ═══ 解析 ═══
    subgraph PARSE["🔧 解析与加载  loader.py → load_data()"]
        P1["1. :: 分隔符 + latin-1 解码"]
        P2["2. 列名映射: UserID→user_id, MovieID→item_id, ..."]
        P3["3. 年份提取: regex \\( (\\d{4}) \\)"]
        P4["4. IMDb 融合: enrich_items_with_imdb()"]
    end

    RAT --> P1
    MOV --> P1
    USR --> P1
    P1 --> P2 --> P3 --> P4

    %% ═══ IMDb 融合 ═══
    subgraph FUSE["🔧 IMDb 多源融合  imdb.py"]
        F1["读取 imdb_cache.json → 按 item_id 查询"]
        F2["写入 imdb_genres 列 + runtime 列"]
        F3["genres 并集: split('|') + split(',') → union → sort → join('|')"]
        F1 --> F2 --> F3
    end

    P4 --> F1
    CACHE -.-> F1

    %% ═══ 基础 Dataset ═══
    F3 --> BASE
    BASE[("📋 基础 Dataset<br/>ratings: 1,000,209 条<br/>items: 3,883 部<br/>users: 6,040 人")]

    %% ═══ 校验 + 清洗 + 特征 ═══
    BASE --> VAL["✅ 契约校验  schema.py<br/>Dataset.validate()"]
    VAL --> CLN["🧹 数据清洗  cleaning.py<br/>去重 / 缺失值 / 低质量标记"]
    CLN --> FEAT["🔢 特征工程  features.py<br/>类型多热矩阵 / 用户画像 / 物品统计"]

    %% ═══ 最终产物 ═══
    FEAT --> FINAL[("🎯 最终 Dataset (含 quality meta)<br/>+ genre_matrix (3883×23)<br/>+ user_features (6040×12)<br/>+ item_stats (3706×6)")]

    %% ═══ 划分 ═══
    subgraph SPLIT["📊 数据划分  loader.py"]
        S1["train_test_split_by_time()<br/>每用户最近 20% → test"]
        S2["train_test_split_with_cold_users()<br/>10% 用户全部行为 → cold_user_test"]
        S3["train_test_split_with_cold_items()<br/>低评分物品 → cold_item_test"]
    end

    FINAL --> S1
    FINAL --> S2
    FINAL --> S3

    S1 --> O1[("Split (常规)<br/>train: 724,752 条<br/>test: 178,521 条")]
    S2 --> O2[("Split (冷用户)<br/>train + test<br/>+ cold_user_test")]
    S3 --> O3[("Split (冷物品)<br/>train + test<br/>+ cold_item_test")]

    O1 --> MOD["models/ 模型训练层"]
    O2 --> MOD
    O3 --> MOD

    %% ═══ 样式 ═══
    classDef src fill:#f5e6c8,stroke:#b8860b,stroke-width:2px
    classDef raw fill:#f5f0e8,stroke:#b7614a,stroke-width:2px
    classDef proc fill:#e8e0d5,stroke:#78716c,stroke-width:1px
    classDef cache fill:#e0d5c5,stroke:#b8860b,stroke-width:1px,stroke-dasharray:5
    classDef ds fill:#d5cfc6,stroke:#1c1917,stroke-width:2px
    classDef final fill:#d4c8b0,stroke:#b8860b,stroke-width:3px
    classDef out fill:#e0d5c5,stroke:#78716c,stroke-width:2px

    class ML,IMDB src
    class RAT,MOV,USR raw
    class E1,E2,E3,E4,E5,P1,P2,P3,P4,F1,F2,F3,VAL,CLN,FEAT proc
    class CACHE cache
    class BASE ds
    class FINAL final
    class O1,O2,O3 out
```

---

## 简化血缘 (横向, 适合 PPT 嵌入)

```mermaid
flowchart LR
    A["MovieLens 1M<br/>ratings + movies + users"] --> B["load_data()<br/>解析 / 列名映射 / 年份提取"]
    B --> C["enrich_items_with_imdb()<br/>genres 并集融合"]
    C --> D[".validate()<br/>契约校验"]
    D --> E["cleaning.py<br/>去重 / 缺失值 / 标记"]
    E --> F["features.py<br/>类型矩阵 / 用户画像"]
    F --> G["split 划分<br/>时间留出 / 冷启动"]
    G --> H["models/"]

    I["IMDb<br/>title.basics.tsv.gz"] --> J["build_imdb_cache.py<br/>ETL → imdb_cache.json"]
    J -.-> C

    style A fill:#f5e6c8,stroke:#b8860b
    style I fill:#f5e6c8,stroke:#b8860b
    style D fill:#d4c8b0,stroke:#b8860b
    style H fill:#d4c8b0,stroke:#b8860b
```

---

## 字段级血缘追溯

```mermaid
flowchart LR
    subgraph SOURCE["原始字段"]
        S1["ratings.dat::UserID"]
        S2["ratings.dat::MovieID"]
        S3["ratings.dat::Rating"]
        S4["ratings.dat::Timestamp"]
        S5["movies.dat::Title"]
        S6["movies.dat::Genres"]
        S7["users.dat::Gender"]
        S8["users.dat::Age"]
        S9["IMDb::genres"]
        S10["IMDb::runtimeMinutes"]
    end

    subgraph TARGET["最终字段"]
        T1["ratings.user_id"]
        T2["ratings.item_id"]
        T3["ratings.rating"]
        T4["ratings.timestamp"]
        T5["items.title"]
        T6["items.release_year  ⇦ regex('(\\\\d{4})')"]
        T7["items.genres  ⇦ ML ∪ IMDb 并集"]
        T8["items.imdb_genres"]
        T9["items.runtime"]
        T10["users.gender"]
        T11["users.age"]
    end

    S1 --> T1
    S2 --> T2
    S3 --> T3
    S4 --> T4
    S5 --> T5
    S5 --> T6
    S6 --> T7
    S9 --> T7
    S9 --> T8
    S10 --> T9
    S7 --> T10
    S8 --> T11

    style S9 fill:#f5e6c8,stroke:#b8860b
    style S10 fill:#f5e6c8,stroke:#b8860b
    style T6 fill:#d4c8b0,stroke:#b8860b
    style T7 fill:#d4c8b0,stroke:#b8860b
```

**说明**: 金色边框 = IMDb 来源字段 / 转换后字段; 虚线 = 多源并集。
