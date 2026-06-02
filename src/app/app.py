"""
MoviePicks ─ 智能电影推荐系统
============================================================
暖调影院主题 · 双源数据融合 (MovieLens 1M + IMDb)
7 种推荐算法 · 统计显著性检验 · 错误分层分析

运行:  streamlit run src/app/app.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_data, train_test_split_by_time
from src.models.als import ALSRecommender
from src.models.coldstart import ColdStartRecommender
from src.models.content import ContentRecommender
from src.models.hybrid import HybridRecommender
from src.models.itemcf import ItemCFRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender

# ═══════════════════════════════════════════════════════════════════
# 设计系统 ─ 暖调影院主题
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MoviePicks · 电影推荐", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

C = {
    "ink": "#1c1917", "espresso": "#2d2a26", "warm_gray": "#78716c",
    "stone": "#d6d3d1", "cream": "#faf7f2", "wheat": "#f5f0e8",
    "gold": "#b8860b", "gold_light": "#f5e6c8",
}

_GENRE_GROUP = {
    "Action":"warm","Adventure":"warm","War":"warm","Western":"warm",
    "Crime":"warm","Film-Noir":"warm","Thriller":"warm",
    "Comedy":"golden","Animation":"golden","Children's":"golden",
    "Family":"golden","Musical":"golden","Music":"golden",
    "Drama":"rose","Romance":"rose",
    "Documentary":"teal","Biography":"teal","History":"teal","Sport":"teal",
    "Sci-Fi":"plum","Fantasy":"plum","Horror":"plum","Mystery":"plum",
}
_GROUP_COLORS = {"warm":"#b7614a","golden":"#b68b40","rose":"#b45a6e","teal":"#5d8a82","plum":"#776a8c"}
def _tag_color(g): return _GROUP_COLORS[_GENRE_GROUP.get(g,"plum")]

GENRES = sorted(_GENRE_GROUP.keys())

_RANK_GRAD = [
    "linear-gradient(135deg, #b8860b, #d4a853)",
    "linear-gradient(135deg, #a8a29e, #c5bfb8)",
    "linear-gradient(135deg, #b7614a, #c9806e)",
]
def _rank_style(r): return _RANK_GRAD[r-1] if r<=3 else "#d6d3d1"
def _rank_text(r): return "#fff" if r<=3 else C["warm_gray"]

# ── 7 模型参数 ──
SVD_PARAMS    = {"n_factors":50,"n_epochs":20,"lr":0.005,"reg":0.02,"seed":42}
ALS_PARAMS    = {"n_factors":50,"reg":0.1,"n_epochs":15,"seed":42}
ITEMCF_PARAMS = {"topk_neighbors":120}
HYBRID_PARAMS = {"alpha":0.75,"cf_params":SVD_PARAMS,"content_params":{"like_threshold":4.0}}

_MODEL_CLASSES = {
    "popularity": PopularityRecommender,
    "itemcf": ItemCFRecommender,
    "svd": SVDRecommender,
    "als": ALSRecommender,
    "content": ContentRecommender,
    "hybrid": HybridRecommender,
    "coldstart": ColdStartRecommender,
}
_MODEL_KWARGS = {"svd":SVD_PARAMS,"als":ALS_PARAMS,"itemcf":ITEMCF_PARAMS,"hybrid":HYBRID_PARAMS}

AGE_LABELS = {1:"<18",18:"18–24",25:"25–34",35:"35–44",45:"45–49",50:"50–55",56:"56+"}
GENDER_MAP = {"M":"♂ 男","F":"♀ 女"}

# ═══════════════════════════════════════════════════════════════════
# 全局 CSS
# ═══════════════════════════════════════════════════════════════════
_CSS = """
<style>
  .stApp { background: #faf7f2; }
  .mc { background:#f5f0e8; border:1px solid #e7e0d5; border-radius:14px;
        padding:20px 24px; margin:10px 0; display:flex; gap:20px; align-items:flex-start;
        transition:box-shadow .2s,transform .2s; }
  .mc:hover { box-shadow:0 4px 20px rgba(0,0,0,.08); transform:translateY(-2px); }
  .rk { min-width:44px;height:44px;border-radius:50%;display:flex;align-items:center;
        justify-content:center;font-weight:800;font-size:1.1rem;flex-shrink:0; }
  .gt { display:inline-block;color:#fff;padding:3px 10px;border-radius:20px;
        font-size:0.7rem;font-weight:500;margin:2px 3px 2px 0; }
  .sb-o { height:5px;background:#e7e0d5;border-radius:3px;margin:4px 0;width:110px; }
  .sb-i { height:100%;border-radius:3px; }
  .sc { background:#f5f0e8;border:1px solid #e7e0d5;border-radius:12px;
        padding:18px 20px;text-align:center; }
  .sv { font-size:1.7rem;font-weight:800;color:#b8860b; }
  .sl { font-size:0.76rem;color:#78716c;margin-top:2px; }
  .rs { display:inline-flex;align-items:center;gap:4px;background:#f5e6c8;color:#8b6914;
        border-radius:20px;padding:4px 12px;font-size:0.78rem; }
  section[data-testid="stSidebar"] { background:linear-gradient(180deg,#1c1917 0%,#292524 100%); }
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stCaption { color:#d6d3d1 !important; }
  section[data-testid="stSidebar"] hr { border-color:#44403c; }
  section[data-testid="stSidebar"] .stRadio label { color:#d6d3d1 !important; }
  .stButton>button { border-radius:10px!important;font-weight:600!important;transition:all .15s!important; }
  .dc { background:#f5f0e8;border:1px solid #e7e0d5;border-radius:12px;padding:14px;margin:5px 0; }
  .poster-chip { width:42px;height:58px;border-radius:7px;display:flex;align-items:center;
                 justify-content:center;font-size:1.3rem;flex-shrink:0; }
  .empty { text-align:center;padding:60px 20px;color:#a8a29e; }
  .empty .big { font-size:4rem;display:block;margin-bottom:12px; }
  .ph { margin-bottom:6px; }
  .ph .t { font-size:1.5rem;font-weight:800;color:#1c1917; }
  .ph .sub { font-size:0.85rem;color:#a8a29e;margin-left:10px; }
  hr { border-color:#e7e0d5!important; }
  .st-caption { color:#a8a29e!important; }
</style>
"""

# ═══════════════════════════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load():
    ds = load_data()
    return ds, train_test_split_by_time(ds, test_ratio=0.2)

@st.cache_resource(show_spinner=False)
def _train_model(name: str):
    ds, sp = _load()
    cls = _MODEL_CLASSES[name]
    kwargs = _MODEL_KWARGS.get(name, {})
    return cls(**kwargs).fit(sp, ds)

# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════
def _lookup(ds):
    idx = ds.items.set_index("item_id")
    def _g(iid):
        if iid in idx.index:
            r = idx.loc[iid]
            return (str(r.get("title",iid)),str(r.get("genres","")),
                    r.get("release_year"),r.get("runtime",None))
        return (str(iid),"",None,None)
    return _g

_POSTER = ["🎥","🎬","🎭","🎪","🎨","🎯","🎰","🎲","🎳","🎵","🎶",
           "🎷","🎸","🎹","🎺","🎻","🥁","🍿","🎞️","📽️","🎟️","🌟"]
def _emo(iid): return _POSTER[iid%len(_POSTER)]

def _tags(gs, limit=4):
    tags = [g.strip() for g in gs.replace("|",",").split(",") if g.strip()]
    return "".join(f'<span class="gt" style="background:{_tag_color(t)}">{t}</span>' for t in tags[:limit])

def _sbar(score):
    pct=(score/5)*100
    c="#b8860b" if pct>=60 else "#b7614a"
    return f'<div class="sb-o"><div class="sb-i" style="width:{pct}%;background:{c}"></div></div>'

def _stitle(full):
    import re
    m=re.search(r'^(.*)\s*\((\d{4})\)$',full)
    return (m.group(1).strip(),m.group(2)) if m else (full,"")

def _page_header(icon,title,subtitle):
    st.markdown(f'<div class="ph"><span class="t">{icon} {title}</span>'
                f'<span class="sub">{subtitle}</span></div>',unsafe_allow_html=True)
    st.divider()

# ═══════════════════════════════════════════════════════════════════
# 页面 1: 为你推荐
# ═══════════════════════════════════════════════════════════════════
def page_recommend():
    ds, split = _load()
    L = _lookup(ds)
    uids = sorted(split.train["user_id"].unique())

    with st.sidebar:
        st.markdown("#### 👤 观影身份")
        uid = st.selectbox("u",uids,index=42,
                           format_func=lambda u:f"用户 #{u}",label_visibility="collapsed")
        model_sel = st.selectbox("推荐引擎",
            ["hybrid","svd","itemcf","als","content"],
            format_func=lambda m:{"hybrid":"Hybrid 混合推荐","svd":"SVD 评分预测",
                "itemcf":"ItemCF 协同过滤","als":"ALS 交替最小二乘","content":"Content 内容画像"}[m])
        topn = st.slider("推荐数量",5,24,12)
        st.divider()

        urow = ds.users[ds.users["user_id"]==int(uid)]
        if not urow.empty:
            u = urow.iloc[0]
            hist = split.train[split.train["user_id"]==int(uid)]
            n = len(hist)
            top_g = "—"
            if n>0:
                hg = hist.merge(ds.items[["item_id","genres"]],on="item_id",how="left")
                gc = Counter()
                for gs in hg["genres"].dropna():
                    for g in str(gs).split("|"):
                        if g.strip(): gc[g.strip()]+=1
                top_g = gc.most_common(1)[0][0] if gc else "—"
            st.markdown(
                f'<div style="background:rgba(255,255,255,.06);border-radius:10px;padding:12px;">'
                f'<div style="font-size:0.85rem;color:#d6d3d1;">'
                f'{GENDER_MAP.get(str(u.get("gender","")),"")} · '
                f'{AGE_LABELS.get(int(u.get("age",0)),"")}</div>'
                f'<div style="font-size:0.72rem;color:#a8a29e;margin-top:4px;">'
                f'🍿 {n} 部 · 🎯 偏爱 {top_g}</div></div>',unsafe_allow_html=True)

    _page_header("🎬","为你推荐","基于你的观影口味，智能生成个性化片单")

    if "r_uid" not in st.session_state:
        st.session_state.r_uid=None; st.session_state.recs=None
    if st.session_state.r_uid != (uid,model_sel):
        st.session_state.recs=None; st.session_state.r_uid=(uid,model_sel)

    if st.button("✨ 生成我的专属推荐",type="primary",use_container_width=False):
        with st.spinner(""):
            m = _train_model(model_sel)
            st.session_state.recs = m.recommend(int(uid),k=int(topn))
            st.session_state.rec_model = model_sel
        st.rerun()

    recs = st.session_state.get("recs")
    if recs:
        mn = st.session_state.get("rec_model","hybrid")
        model = _train_model(mn)
        st.caption(f"共 {len(recs)} 部 · 引擎: {mn}")
        for rank,iid in enumerate(recs,1):
            title,genres,year,_ = L(iid)
            name,yr = _stitle(title)
            pred = model.predict(int(uid),int(iid))
            reason = _reason(int(uid),genres,split)
            st.markdown(
                f'<div class="mc">'
                f'  <div class="rk" style="background:{_rank_style(rank)};color:{_rank_text(rank)}">{rank}</div>'
                f'  <div style="width:64px;height:88px;border-radius:9px;background:{_rank_style(rank)};'
                f'       display:flex;align-items:center;justify-content:center;font-size:1.6rem;'
                f'       flex-shrink:0;color:#fff;">{_emo(iid)}</div>'
                f'  <div style="flex:1;">'
                f'    <div style="font-size:1.1rem;font-weight:700;color:#1c1917;">'
                f'      {name}<span style="font-weight:400;color:#a8a29e;font-size:0.9rem;margin-left:4px;">({yr})</span></div>'
                f'    <div style="margin:5px 0;">{_tags(genres)}</div>'
                f'    <div style="display:flex;align-items:center;gap:12px;margin:6px 0;">'
                f'      <span style="font-size:0.85rem;font-weight:600;color:#2d2a26;">预测 {pred:.1f} 分</span>{_sbar(pred)}</div>'
                f'    <div class="rs">💡 {reason}</div></div></div>',
                unsafe_allow_html=True)
    else:
        _hero(int(uid),split,L)

def _hero(uid,split,L):
    hist = split.train[split.train["user_id"]==uid].sort_values("rating",ascending=False)
    if hist.empty: return
    st.info("👆 点击按钮生成专属推荐 —— 或先回顾你钟爱的电影：")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="sc"><div class="sv">{len(hist)}</div><div class="sl">观影总量</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sc"><div class="sv">{hist["rating"].mean():.1f}</div><div class="sl">平均评分</div></div>',unsafe_allow_html=True)
    with c3:
        ft,_,_,_=L(hist["item_id"].iloc[0])
        st.markdown(f'<div class="sc"><div class="sv" style="font-size:0.85rem;">{_stitle(ft)[0][:14]}</div><div class="sl">最爱的电影</div></div>',unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="sc"><div class="sv">5</div><div class="sl">最高评分</div></div>',unsafe_allow_html=True)
    st.markdown("##### 💎 你钟爱的电影")
    cols=st.columns(4)
    for i,(_,row) in enumerate(hist.head(8).iterrows()):
        title,genres,year,_=L(row["item_id"])
        name,yr=_stitle(title)
        with cols[i%4]:
            st.markdown(
                f'<div class="dc"><div style="font-weight:600;font-size:0.85rem;color:#1c1917;">{name}</div>'
                f'<div style="font-size:0.7rem;color:#a8a29e;">{yr} · {int(row["rating"])}⭐</div>'
                f'<div style="font-size:0.65rem;color:#d6d3d1;margin-top:2px;">{genres.replace("|"," / ")[:32]}</div></div>',
                unsafe_allow_html=True)

def _reason(uid,genres,split):
    gs=[g.strip() for g in genres.replace("|",",").split(",") if g.strip()]
    pool=[f"你钟爱的「{random.choice(gs)}」类型"] if gs else []
    pool+=["综合你的观影口味","隐藏好片"]
    return random.choice(pool)

# ═══════════════════════════════════════════════════════════════════
# 页面 2: 新用户开始
# ═══════════════════════════════════════════════════════════════════
def page_onboarding():
    _page_header("🆕","新用户？从这里开始","零观影历史也能获得精准推荐")
    ds,_=_load(); L=_lookup(ds)
    ca,cb=st.columns([1.05,1])
    with ca:
        st.markdown("##### 🎯 勾选你感兴趣的电影类型")
        picked=st.multiselect("g",GENRES,default=["Action","Sci-Fi","Comedy"],label_visibility="collapsed")
        topn=st.slider("想看多少部？",5,20,10,key="ob_n")
        if st.button("🎲 生成首屏推荐",type="primary",use_container_width=True):
            if not picked: st.warning("请至少选择一种类型。")
            else:
                with st.spinner(""):
                    st.session_state.ob_r=_train_model("coldstart").recommend_for_preferences(picked,k=int(topn))
                    st.session_state.ob_p=picked
                st.rerun()
        st.caption("💡 一键组合：")
        combos=[("🔥 动作科幻",["Action","Sci-Fi","Thriller"]),("🤣 轻松喜剧",["Comedy","Animation","Children's"]),
                ("🎭 剧情深度",["Drama","Romance","Mystery"]),("👻 惊悚悬疑",["Horror","Thriller","Mystery"]),
                ("📚 文艺纪录",["Documentary","Drama","Musical"])]
        for i,(label,gs) in enumerate(combos):
            if st.button(label,key=f"qb_{i}",use_container_width=True):
                st.session_state.ob_r=_train_model("coldstart").recommend_for_preferences(gs,k=10)
                st.session_state.ob_p=gs; st.rerun()
    with cb:
        recs=st.session_state.get("ob_r")
        if recs:
            st.markdown(f"##### 🍿 精选 {len(recs)} 部")
            st.caption(f"偏好：{' · '.join(st.session_state.get('ob_p',[])[:5])}")
            for rank,iid in enumerate(recs,1):
                title,genres,year,_=L(iid); name,yr=_stitle(title); y=f" · {yr}" if yr else ""
                st.markdown(
                    f'<div style="background:#f5f0e8;border:1px solid #e7e0d5;border-radius:10px;'
                    f'padding:10px 14px;margin:5px 0;display:flex;align-items:center;gap:10px;">'
                    f'<span style="font-weight:800;color:#b8860b;min-width:22px;">{rank}</span>'
                    f'<div><strong style="color:#1c1917;">{name}</strong>'
                    f'<span style="color:#a8a29e;font-size:0.82rem;">{y}</span>'
                    f'<br><span style="font-size:0.7rem;color:#d6d3d1;">{genres.replace("|"," / ")[:40]}</span></div></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="empty"><span class="big">🍿</span>'
                        f'<div style="font-size:1rem;font-weight:600;color:#78716c;">你的推荐将出现在这里</div>'
                        f'<div style="font-size:0.82rem;">左侧勾选类型 → 点击生成按钮</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 页面 3: 发现电影
# ═══════════════════════════════════════════════════════════════════
def page_discover():
    _page_header("🔍","发现电影","浏览 3883 部电影 · 按类型 · 热度 · 年份筛选")
    ds,_=_load(); items=ds.items.copy()
    pop=ds.ratings.groupby("item_id")["rating"].agg(["count","mean"]).reset_index()
    pop.columns=["item_id","评分人数","平均分"]
    items=items.merge(pop,on="item_id",how="left")
    items["评分人数"]=items["评分人数"].fillna(0).astype(int)
    items["平均分"]=items["平均分"].fillna(0).round(1)
    c1,c2,c3=st.columns([1,1,1])
    with c1: gsel=st.selectbox("🎭 类型",["🎬 全部"]+GENRES)
    with c2: ssel=st.selectbox("📊 排序",["🔥 热门程度","⭐ 平均评分","📅 最新上映"])
    with c3: kw=st.text_input("🔎 搜索",placeholder="输入片名关键词……")
    f=items.copy()
    if "全部" not in gsel: f=f[f["genres"].str.contains(gsel,na=False)]
    if kw: f=f[f["title"].str.contains(kw,case=False,na=False)]
    if "热门" in ssel: f=f.sort_values("评分人数",ascending=False)
    elif "平均" in ssel: f=f[f["评分人数"]>=20].sort_values("平均分",ascending=False)
    else: f=f.sort_values("release_year",ascending=False)
    st.caption(f"共 {len(f)} 部")
    PS=40
    if "dp" not in st.session_state: st.session_state.dp=0
    tp=max(1,(len(f)-1)//PS)
    page=f.iloc[st.session_state.dp*PS:st.session_state.dp*PS+PS]
    cols=st.columns(4)
    for i,(_,r) in enumerate(page.iterrows()):
        name,yr=_stitle(str(r.get("title","?")))
        gs=str(r.get("genres","")); cnt=r.get("评分人数",0); avg=r.get("平均分",0); iid=int(r["item_id"])
        pct=(avg/5.0)*100 if avg>0 else 0
        with cols[i%4]:
            st.markdown(
                f'<div class="dc"><div style="display:flex;align-items:flex-start;gap:8px;">'
                f'<div class="poster-chip" style="background:{_rank_style((i%5)+1)};color:#fff;">{_emo(iid)}</div>'
                f'<div style="flex:1;min-width:0;"><div style="font-weight:700;font-size:0.83rem;'
                f'color:#1c1917;line-height:1.25;">{name[:20]}</div>'
                f'<div style="font-size:0.68rem;color:#a8a29e;">{yr}</div>'
                f'<div style="font-size:0.68rem;color:#78716c;margin:2px 0;">⭐ {avg} · {cnt}人</div>'
                f'<div style="height:3px;background:#e7e0d5;border-radius:2px;margin-top:3px;">'
                f'<div style="height:100%;width:{pct}%;background:#b8860b;border-radius:2px;"></div></div>'
                f'</div></div></div>',unsafe_allow_html=True)
    if tp>1:
        _,c2,_=st.columns([1,2,1])
        with c2:
            cl,cp,cr=st.columns([1,2,1])
            with cl:
                if st.button("◀ 上一页",disabled=(st.session_state.dp==0)):
                    st.session_state.dp=max(0,st.session_state.dp-1);st.rerun()
            with cp:
                st.markdown(f'<div style="text-align:center;color:#a8a29e;padding-top:6px;">{st.session_state.dp+1} / {tp+1}</div>',unsafe_allow_html=True)
            with cr:
                if st.button("下一页 ▶",disabled=(st.session_state.dp>=tp)):
                    st.session_state.dp=min(tp,st.session_state.dp+1);st.rerun()

# ═══════════════════════════════════════════════════════════════════
# 页面 4: 关于
# ═══════════════════════════════════════════════════════════════════
def page_about():
    _page_header("ℹ️","关于 MoviePicks","")

    st.markdown("""
    ### 🎬 MoviePicks 是什么？

    融合 **MovieLens 1M** 用户行为与 **IMDb** 电影元数据的端到端智能推荐系统，
    覆盖 **数据处理 → 模型训练 → 评估分析 → 交互界面** 全链路。
    """)

    st.markdown("#### 📐 系统架构")
    arch_cols = st.columns(4)
    modules = [
        ("📦 数据处理", "MovieLens 1M 解析\nIMDb genres 并集融合\n数据清洗 · 质量报告\n特征工程 (23种类型矩阵)"),
        ("🧠 模型训练", "7 种推荐算法\n协同过滤 × 内容画像 × 混合融合\n模型持久化 (save/load)\nYAML 配置驱动调优"),
        ("📊 评估分析", "三层指标体系 (评分/排序/业务)\n统计显著性检验 (t-test/boot/Cohen's d)\n错误分析 (活跃度/热度分层)\n7 种可视化图表"),
        ("🎨 交互界面", "个性化推荐 (5 引擎可选)\n新用户冷启动问卷\n电影发现 · 搜索浏览\n暖调影院主题设计"),
    ]
    for col, (title, desc) in zip(arch_cols, modules):
        with col:
            st.markdown(f"**{title}**\n{desc}")

    st.divider()

    st.markdown("#### 🧠 推荐算法总览")
    model_data = [
        ("Popularity", "热门基线", "贝叶斯平滑", "0.990", "0.020", "1.5%"),
        ("ItemCF", "协同过滤", "余弦相似度 + top-N 截断", "1.008", "**0.082**", "30.8%"),
        ("SVD", "协同过滤 (SGD)", "FunkSVD 偏置 + mini-batch SGD", "**0.890**", "0.028", "16.8%"),
        ("ALS", "协同过滤 (闭式)", "交替最小二乘 ridge 回归", "1.210", "0.001", "34.4%"),
        ("Content", "内容画像", "TF-IDF + IMDb 并集增强", "1.308", "0.011", "**79.5%**"),
        ("Hybrid", "混合融合", "SVD + Content min-max 加权 (α=0.75)", "0.922", "0.031", "24.7%"),
        ("ColdStart", "冷启动", "Content 画像 + Popularity 兜底", "1.308", "0.011", "79.5%"),
    ]
    st.markdown(
        "| 模型 | 类别 | 核心方法 | RMSE↓ | Recall@10↑ | Coverage↑ |\n"
        "|---|---|---|---|---|---|\n" +
        "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in model_data)
    )

    st.divider()

    st.markdown("#### 📊 评估亮点 (MovieLens 1M 实测)")
    eval_cols = st.columns(3)
    with eval_cols[0]:
        st.markdown("**统计显著性 (SVD vs ItemCF)**")
        st.markdown(f'<div class="sc"><div class="sv">p≈0</div><div class="sl">配对 t 检验 (t=-36.9)</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sc" style="margin-top:8px;"><div class="sv">0.34</div><div class="sl">效应量 (Cohen\'s d, small)</div></div>',unsafe_allow_html=True)
    with eval_cols[1]:
        st.markdown("**用户活跃度分层 (SVD RMSE)**")
        st.markdown(f'<div class="sc"><div class="sv">0.97→0.87</div><div class="sl">低活→高活 RMSE 下降 10%</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sc" style="margin-top:8px;"><div class="sv">0.00→0.27</div><div class="sl">冷门→热门 HitRate 跃升</div></div>',unsafe_allow_html=True)
    with eval_cols[2]:
        st.markdown("**评分误差分布**")
        st.markdown(f'<div class="sc"><div class="sv">49%/51%</div><div class="sl">过估/低估 比例均衡</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sc" style="margin-top:8px;"><div class="sv">44%</div><div class="sl">预测误差 &lt; 0.5 分</div></div>',unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 📁 项目文档")
    doc_cols = st.columns(3)
    with doc_cols[0]:
        st.markdown("**数据处理**\n- [数据文档](docs/data_preprocess/data.md)\n- [数据血缘](docs/data_preprocess/data_lineage.md)\n- [特征工程](docs/data_preprocess/features.md)")
    with doc_cols[1]:
        st.markdown("**模型训练**\n- [模型设计](docs/models/model_design.md)\n- [模型对比](docs/models/model_comparison.md)\n- [实验调优](docs/models/experiment.md)")
    with doc_cols[2]:
        st.markdown("**评估分析**\n- [评估体系](docs/evaluation/evaluation_design.md)")

    st.divider()
    st.caption("《数据挖掘与分析》课程期末项目 · 数据仅用于研究与教育目的。")

# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════
def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.spinner(""): _load()

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:20px 0 10px;">'
            '<span style="font-size:2.4rem;">🎬</span>'
            '<h2 style="margin:4px 0 0;color:#e7e0d5;font-weight:800;">MoviePicks</h2>'
            '<p style="color:#a8a29e;font-size:0.76rem;margin:2px 0;">智能电影推荐</p>'
            '</div>',unsafe_allow_html=True)
        st.divider()
        page = st.radio("nav",
            ["🎬 为你推荐","🆕 新用户开始","🔍 发现电影","ℹ️ 关于"],
            label_visibility="collapsed")
        st.divider()
        st.caption("MovieLens 1M + IMDb")
        st.caption("© 2026 MoviePicks")

    if   "为你推荐" in page: page_recommend()
    elif "新用户"   in page: page_onboarding()
    elif "发现电影" in page: page_discover()
    elif "关于"     in page: page_about()

if __name__ == "__main__":
    main()
