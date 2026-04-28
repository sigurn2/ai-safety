"""
Streamlit 应用入口：AI 治理监测演示看板（汇报版）。

功能：四 Tab 看板（监测 / 情报 / 深度调研 / 系统）；缓存全部查询；
     侧边栏受密码保护的操作区供现场演示触发同步与 Agent 侦察。
输入：MySQL（articles / article_extractions）供看板只读；DB_PATH 仅 Agent 演示写 SQLite。
输出：页面渲染；操作区副作用：网络请求 + MySQL（卫报同步）或 SQLite（Agent）。
上下游：依赖 core.mysql_dashboard、core.db（init/save_incident）、crawler.orchestrator、crawler.agentic_crawl；
        由 systemd 在服务器持续运行，Nginx 反代对外暴露。
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import (
    API_KEY,
    BASE_URL,
    DB_PATH,
    GUARDIAN_API_KEY,
    LLM_MODEL,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PORT,
)
from core.db import coerce_risk_domain, incident_from_extraction, init_db, save_incident
from core.llm_client import OpenAICompatibleBackend
from core.mysql_dashboard import (
    fetch_dashboard_all_rows,
    fetch_dashboard_latest_rows,
    get_dashboard_keywords_df,
    get_dashboard_stats,
    get_dashboard_taxonomy_df,
)
from core.mysql_db import get_research_report_by_id, list_research_reports, save_research_report
from engine.rag_ingestion.hybrid_retrieval import evidence_hits_to_report_sources, hybrid_retrieve
from engine.research_report import generate_research_report_markdown
from models.schema import RISK_DOMAIN_CHOICES

# Windows 下 Playwright 子进程兼容
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 环形图配色（与页面深色主题一致）
_DONUT_COLORS = (
    "#4f8ef7",
    "#3db88a",
    "#a78bfa",
    "#f0ab43",
    "#e879a8",
    "#5eb3f6",
    "#7dd3c0",
    "#c4b5fd",
    "#fbbf24",
    "#fb923c",
    "#38bdf8",
    "#94a3b8",
)


def _donut_color_list(n: int) -> list[str]:
    base = list(_DONUT_COLORS)
    out: list[str] = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


def _fig_domain_donut(labels: list[str], values: list[int]) -> go.Figure:
    n = len(labels)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.54,
                pull=[0.025] * n,
                marker=dict(
                    colors=_donut_color_list(n),
                    line=dict(color="#0f1424", width=2),
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(color="#e8eaf6", size=13),
                insidetextorientation="horizontal",
                hovertemplate="<b>%{label}</b><br>篇数: %{value}<br>占比: %{percent}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            x=1.02,
            xanchor="left",
            font=dict(color="#a8b3cf", size=11),
            bgcolor="rgba(0,0,0,0)",
            itemwidth=30,
        ),
        margin=dict(t=20, b=20, l=20, r=190),
        height=360,
    )
    return fig


def _fig_subdomain_donut(labels: list[str], values: list[int]) -> go.Figure:
    n = len(labels)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.54,
                pull=[0.018] * n,
                marker=dict(
                    colors=_donut_color_list(n),
                    line=dict(color="#0f1424", width=2),
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(color="#e8eaf6", size=11),
                insidetextorientation="horizontal",
                hovertemplate="<b>%{label}</b><br>篇数: %{value}<br>占比: %{percent}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            x=1.02,
            xanchor="left",
            font=dict(color="#a8b3cf", size=9),
            bgcolor="rgba(0,0,0,0)",
            itemwidth=30,
        ),
        margin=dict(t=20, b=20, l=20, r=240),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# 缓存包装：所有只读查询加 2 分钟缓存，翻 Tab 不重查库
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def _cached_stats() -> Tuple[int, int, int]:
    """功能：缓存版 MySQL 汇总；输出：(extractions 数, 标签去重数, 主域×子域组合种数)。"""
    try:
        return get_dashboard_stats()
    except Exception:
        return 0, 0, 0


@st.cache_data(ttl=120)
def _cached_taxonomy() -> pd.DataFrame:
    """功能：缓存版主域×子域频次（MySQL JSON 展开聚合）。"""
    try:
        return get_dashboard_taxonomy_df()
    except Exception:
        return pd.DataFrame(columns=["domain", "subdomain", "tax_count", "first_seen"])


@st.cache_data(ttl=120)
def _cached_keywords() -> pd.DataFrame:
    """功能：缓存版 tags_raw 聚合高频词（Top 60）。"""
    try:
        return get_dashboard_keywords_df()
    except Exception:
        return pd.DataFrame(columns=["keyword", "count"])


@st.cache_data(ttl=60)
def _cached_latest_incidents(limit: int = 20) -> pd.DataFrame:
    """功能：缓存最新情报列表（MySQL）；输入：limit；输出：DataFrame。"""
    try:
        return fetch_dashboard_latest_rows(limit)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _cached_all_incidents() -> pd.DataFrame:
    """功能：缓存全量情报供详情 Tab 筛选（MySQL）。"""
    try:
        return fetch_dashboard_all_rows()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def _cached_research_report_list(limit: int = 25) -> pd.DataFrame:
    """近期深度调研报告列表（MySQL research_reports）。"""
    try:
        rows = list_research_reports(limit=limit)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 密码验证：从环境变量读取演示密码；未设置则关闭保护
# ---------------------------------------------------------------------------

def _demo_unlocked() -> bool:
    """
    功能：校验侧边栏密码输入，未设 DEMO_PASSWORD 时始终返回 True。
    输入：st.session_state 中的 demo_pwd 字段。
    输出：布尔；无 IO。
    """
    required = os.getenv("DEMO_PASSWORD", "").strip()
    if not required:
        return True
    entered = st.session_state.get("demo_pwd", "")
    return entered == required


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------

def main() -> None:
    """
    功能：配置页面、渲染三 Tab 看板与侧边栏演示操作区。
    输入：无参数；依赖 Streamlit session 与环境变量。
    输出：无；副作用：init_db（Agent SQLite）；只读看板查 MySQL。
    """
    st.set_page_config(
        page_title="全球 AI 治理监测系统",
        layout="wide",
        page_icon="🛡️",
        initial_sidebar_state="collapsed",
    )
    init_db()

    # 全局 CSS：统一卡片与标签样式
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #242b4a 100%);
        border: 1px solid #2a3563;
        border-left: 4px solid #4f8ef7;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 8px;
    }
    .metric-card .label { color: #8892b0; font-size: 13px; margin-bottom: 4px; }
    .metric-card .value { color: #e8eaf6; font-size: 32px; font-weight: 700; line-height: 1; }
    .metric-card .delta { color: #4ade80; font-size: 12px; margin-top: 4px; }
    .tag-chip {
        background: #1e2130; color: #7eb8f7; padding: 3px 10px;
        border-radius: 12px; margin: 2px; border: 1px solid #2a3563;
        display: inline-block; font-size: 12px;
    }
    .section-header {
        border-bottom: 2px solid #2a3563;
        padding-bottom: 6px;
        margin-bottom: 16px;
        color: #c7d0e8;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- 标题区 ---
    col_title, col_ts = st.columns([4, 1])
    with col_title:
        st.markdown("## 🛡️ 国际动态监测平台")
        st.caption("基于大语言模型的 AI 安全动态智能感知平台 · 实时追踪监管政策、技术风险与治理事件")
    with col_ts:
        st.caption(f"数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # --- 核心指标（全部来自数据库）---
    total_incidents, total_tags, taxonomy_kinds = _cached_stats()
    kw_df = _cached_keywords()
    kw_total = len(kw_df) if not kw_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("识别风险情报", total_incidents, help="已入库的 AI 治理/安全事件总数")
    with c2:
        st.metric("去重关键词总量", total_tags, help="从所有情报标签中提取的独立关键词数")
    with c3:
        st.metric("风险子域种数", taxonomy_kinds, help="动态演化的风险分类体系中不同子域数量")
    with c4:
        st.metric("自增长词库节点", kw_total, help="系统自动发现并持续追踪的领域术语数量")

    st.divider()

    # --- 四 Tab 看板 ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 监测看板", "📋 情报详情", "📚 深度调研", "⚙️ 系统状态"])

    # ================================================================
    # Tab 1 - 监测看板
    # ================================================================
    with tab1:
        left, right = st.columns([3, 2])

        with left:
            st.markdown('<div class="section-header">📍 最新监测情报</div>', unsafe_allow_html=True)
            df_latest = _cached_latest_incidents(20)
            if not df_latest.empty:
                # 主域缩短显示
                if "主域" in df_latest.columns:
                    df_latest["主域"] = (
                        df_latest["主域"].astype(str)
                        .str.replace(r"\s*\(.+$", "", regex=True)
                        .str.strip()
                    )
                st.dataframe(
                    df_latest.drop(columns=["来源"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )
            else:
                st.info("暂无监测数据，请从演示操作区触发同步。")

            # 三元主域分布
            st.markdown(
                '<div class="section-header" style="margin-top:24px">'
                "🌳 动态风险分类体系（三元主域 → 子域）</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "主域划分对齐 AI 安全与治理领域通行的「意图—来源」三类风险表述，便于与主流政策与学术话语对接；"
                "子域由抽取结果与语料统计动态演化。"
            )
            with st.expander("分类口径与依据（说明）", expanded=False):
                st.markdown(
                    """
**三元主域**对应学界与产业常用的风险分层：**恶意滥用**（Malicious Use）、**意外失效**
（Accidental Failure / 可靠性）、**系统性与伦理风险**（Systemic & Ethical），与 NIST AI RMF、
OECD AI 原则、欧盟《人工智能法案》等国内外治理框架中的风险维度在**语义上可对齐**（非对某一条款的逐字映射）。

**子域**为在各主域下由模型标注、检索增强与词频统计共同沉淀的议题标签，会随监测语料扩充而**自动演化**。
                    """.strip()
                )
            tax_df = _cached_taxonomy()
            if not tax_df.empty:
                dom_cols = st.columns(3)
                for i, domain_label in enumerate(RISK_DOMAIN_CHOICES):
                    short = domain_label.split("(")[0].strip()
                    sub_df = tax_df[tax_df["domain"] == domain_label].head(10)
                    with dom_cols[i]:
                        st.markdown(f"**{short}**")
                        if sub_df.empty:
                            st.caption("—")
                        else:
                            for _, row in sub_df.iterrows():
                                st.caption(f"· {row['subdomain']}（×{int(row['tax_count'])}）")
            else:
                st.caption("子域数据积累中，入库带 risk_subdomain 的情报后自动更新。")

        with right:
            st.markdown('<div class="section-header">📊 风险主域分布</div>', unsafe_allow_html=True)
            tax_df_r = _cached_taxonomy()
            if not tax_df_r.empty:
                domain_agg = tax_df_r.groupby("domain")["tax_count"].sum().reset_index()
                domain_agg["主域"] = (
                    domain_agg["domain"].str.replace(r"\s*\(.+$", "", regex=True).str.strip()
                )
                domain_agg = domain_agg.rename(columns={"tax_count": "情报数"})
                fig_domain = _fig_domain_donut(
                    domain_agg["主域"].tolist(),
                    pd.to_numeric(domain_agg["情报数"], errors="coerce").fillna(0).astype(int).tolist(),
                )
                st.plotly_chart(fig_domain, use_container_width=True)

                st.markdown(
                    '<div class="section-header" style="margin-top:20px">'
                    "🔥 高频风险子域 (Top 8 + 其他)</div>",
                    unsafe_allow_html=True,
                )
                sub_sorted = tax_df_r.sort_values("tax_count", ascending=False).reset_index(drop=True)
                short_dom = sub_sorted["domain"].str.replace(r"\s*\(.+$", "", regex=True).str.strip()
                if len(sub_sorted) > 8:
                    head = sub_sorted.head(8)
                    short_h = short_dom.head(8)
                    labels = (head["subdomain"] + " · " + short_h).tolist()
                    vals = pd.to_numeric(head["tax_count"], errors="coerce").fillna(0).astype(int).tolist()
                    other_count = int(pd.to_numeric(sub_sorted["tax_count"].iloc[8:], errors="coerce").fillna(0).sum())
                    if other_count > 0:
                        labels.append("其他")
                        vals.append(other_count)
                else:
                    labels = (sub_sorted["subdomain"] + " · " + short_dom).tolist()
                    vals = pd.to_numeric(sub_sorted["tax_count"], errors="coerce").fillna(0).astype(int).tolist()
                fig_sub = _fig_subdomain_donut(labels, vals)
                st.plotly_chart(fig_sub, use_container_width=True)
            else:
                st.caption("暂无分类统计数据。")

            # 关键词池
            st.markdown('<div class="section-header" style="margin-top:20px">🧬 自增长关键词池</div>', unsafe_allow_html=True)
            if not kw_df.empty:
                top_kw = kw_df.head(40)
                tag_html = "".join([
                    f'<span class="tag-chip">{row["keyword"]}'
                    f'<span style="opacity:0.5;font-size:10px"> ×{row["count"]}</span></span>'
                    for _, row in top_kw.iterrows()
                ])
                st.markdown(tag_html, unsafe_allow_html=True)
            else:
                st.caption("🌱 词库为空，触发一次同步后自动填充。")

    # ================================================================
    # Tab 2 - 情报详情
    # ================================================================
    with tab2:
        st.markdown('<div class="section-header">📋 全量情报库（可筛选）</div>', unsafe_allow_html=True)
        df_all = _cached_all_incidents()

        if df_all.empty:
            st.info("暂无数据，请先从演示操作区触发同步。")
        else:
            # 筛选条件
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                domains = ["全部"] + list(RISK_DOMAIN_CHOICES)
                sel_domain = st.selectbox("按主域筛选（三元模型）", domains, key="filter_domain")
            with fc2:
                levels = ["全部"] + sorted(df_all["资讯类别"].dropna().unique().tolist())
                sel_level = st.selectbox("按资讯类别筛选", levels, key="filter_level")
            with fc3:
                kw_search = st.text_input("关键词搜索（标题/摘要）", key="kw_search")

            df_view = df_all.copy()
            if sel_domain != "全部":
                df_view = df_view[df_view["主域"].map(lambda x: coerce_risk_domain(str(x))) == sel_domain]
            if sel_level != "全部":
                df_view = df_view[df_view["资讯类别"] == sel_level]
            if kw_search.strip():
                mask = (
                    df_view["标题"].str.contains(kw_search, case=False, na=False)
                    | df_view["摘要"].str.contains(kw_search, case=False, na=False)
                )
                df_view = df_view[mask]

            st.caption(f"共 {len(df_view)} 条情报（全库 {len(df_all)} 条）")
            st.dataframe(
                df_view.drop(columns=["id"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
                height=420,
            )

            # CSV 导出
            csv_bytes = df_view.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 导出当前筛选结果（CSV）",
                data=csv_bytes,
                file_name=f"AI_Governance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        st.divider()

        # 日报生成
        st.markdown('<div class="section-header">📄 自动化监测日报</div>', unsafe_allow_html=True)
        if st.button("📥 一键生成 AI 治理监测日报", key="gen_report"):
            df_report = _cached_latest_incidents(10)
            if not df_report.empty:
                report_md = f"## AI 治理动态监测内参（{datetime.now().strftime('%Y-%m-%d')}）\n\n"
                report_md += "### 一、最新情报摘要\n\n"
                for _, row in df_report.iterrows():
                    ctype = str(row.get("资讯类别", "") or "").strip()
                    dom = str(row.get("主域", "") or "").strip()
                    sub = str(row.get("子域", "") or "").strip()
                    entity = str(row.get("涉及主体", "") or "").strip()
                    tri = f"{dom} / {sub}".strip(" /")
                    report_md += f"- **[{ctype or '—'}]** {row['title']}（涉及主体：{entity or '—'}）"
                    if tri:
                        report_md += f" — 分类：{tri}"
                    report_md += "\n"

                report_md += "\n### 二、新兴术语感知\n\n"
                kw_top = _cached_keywords().head(10)
                if not kw_top.empty:
                    report_md += "- 高频新词：" + "、".join(kw_top["keyword"].tolist()) + "\n"

                report_md += f"\n### 三、系统统计\n\n"
                stats = _cached_stats()
                report_md += (
                    f"- 已监测情报：{stats[0]} 条\n"
                    f"- 风险子域种数：{stats[2]} 种\n"
                    f"- 关键词库节点：{kw_total} 个\n"
                )

                st.code(report_md, language="markdown")
                st.download_button(
                    "下载 Markdown 日报",
                    data=report_md.encode("utf-8"),
                    file_name=f"AI_Governance_Daily_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    key="dl_report",
                )
            else:
                st.warning("数据库暂无数据，请先触发同步。")

    # ================================================================
    # Tab 3 - 问答式深度调研（混合检索 + LLM 报告）
    # ================================================================
    with tab3:
        st.markdown(
            '<div class="section-header">📚 问答式深度调研</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "基于 Chroma 向量 + MySQL 全文（若已迁移）混合检索证据，由大模型生成带引用的 Markdown 报告；"
            "可选择写入 `research_reports` 便于留痕。"
        )
        rq = st.text_area(
            "研究问题",
            height=88,
            placeholder="例如：欧盟 AI 法案执法近期有哪些公开讨论？",
            key="deep_research_question",
        )
        dr1, dr2, dr3 = st.columns(3)
        with dr1:
            dom_opts = ["（不筛选）"] + list(RISK_DOMAIN_CHOICES)
            dr_domain_sel = st.selectbox("主域筛选（可选）", dom_opts, key="dr_domain")
            dr_risk_domain = None if dr_domain_sel.startswith("（") else dr_domain_sel
        with dr2:
            dr_source = st.text_input("信源 source 精确匹配（可选）", "", key="dr_source")
        with dr3:
            dr_top_k = st.slider("纳入证据条数", 6, 32, 16, key="dr_top_k")

        dr_save = st.checkbox("生成后写入 MySQL（research_reports + 引用行）", value=True, key="dr_save")
        dr_preview = st.checkbox("仅检索证据、暂不调用 LLM（调试用）", value=False, key="dr_preview")

        if st.button("🔎 检索并生成报告", type="primary", use_container_width=True, key="dr_run"):
            if not (rq or "").strip():
                st.warning("请先填写研究问题。")
            elif not API_KEY:
                st.error("未配置 DASHSCOPE_API_KEY，无法调用大模型生成报告。")
            else:
                hits = []
                report_md = ""
                retrieve_err: Optional[Exception] = None
                gen_err: Optional[Exception] = None
                with st.spinner("正在混合检索并生成报告（篇幅较长时可能需要 1～3 分钟）…"):
                    try:
                        tk = int(dr_top_k)
                        pool = min(64, max(28, tk * 4))
                        hits = hybrid_retrieve(
                            rq.strip(),
                            top_k=tk,
                            risk_domain=dr_risk_domain,
                            source=(dr_source or "").strip() or None,
                            vector_top_n=pool,
                            sparse_top_n=pool,
                            max_chunks_per_article=3,
                        )
                    except Exception as e:
                        retrieve_err = e
                        hits = []

                    if retrieve_err is None and hits and not dr_preview:
                        try:
                            backend = OpenAICompatibleBackend()
                            report_md = generate_research_report_markdown(
                                rq.strip(),
                                hits,
                                backend=backend,
                                model=LLM_MODEL,
                            )
                        except Exception as e:
                            gen_err = e
                            report_md = ""

                if retrieve_err is not None:
                    st.error(f"检索失败：{type(retrieve_err).__name__}: {retrieve_err}")
                elif hits:
                    st.success(f"已检索 **{len(hits)}** 条证据（RRF 融合后；每篇最多 3 块）。")
                    with st.expander("证据预览", expanded=False):
                        for idx, h in enumerate(hits, 1):
                            prev = (h.chunk_text or "").replace("\n", " ")[:220]
                            st.caption(f"**{idx}.** article_id={h.article_id} rrf={h.rrf_score:.4f} — {prev}…")

                if dr_preview:
                    st.info("已开启「仅检索」：跳过 LLM；取消勾选后可生成完整报告。")
                elif retrieve_err is not None:
                    pass
                elif not hits:
                    st.warning("无命中证据，未生成正文。")
                elif gen_err is not None:
                    st.error(f"报告生成失败：{type(gen_err).__name__}: {gen_err}")
                elif not (report_md or "").strip():
                    st.warning("模型返回为空，请重试或检查 API/模型与上下文长度限制。")
                else:
                    st.markdown(report_md)
                    src_payload = evidence_hits_to_report_sources(hits)
                    filt = {
                        "risk_domain": dr_risk_domain or "",
                        "source": (dr_source or "").strip(),
                        "top_k": int(dr_top_k),
                    }
                    if dr_save:
                        try:
                            rid = save_research_report(
                                rq.strip(),
                                filt,
                                report_md,
                                model_name=LLM_MODEL,
                                sources=src_payload,
                            )
                            st.caption(f"已保存至 MySQL，`research_reports.id` = **{rid}**")
                            st.cache_data.clear()
                        except Exception as e:
                            st.warning(f"报告已展示，但入库失败：{type(e).__name__}: {e}")

                    fn = f"DeepResearch_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                    st.download_button(
                        "下载 Markdown 报告",
                        data=report_md.encode("utf-8"),
                        file_name=fn,
                        mime="text/markdown",
                        key="dr_dl_md",
                    )

        st.divider()
        st.markdown("**近期已保存报告**")
        hist = _cached_research_report_list(30)
        if hist.empty:
            st.caption("暂无历史记录；成功保存后此处刷新可见（约 30s 内缓存）。")
        else:
            records = hist.to_dict("records")
            pick_i = st.selectbox(
                "选择一条查看",
                range(len(records)),
                format_func=lambda i: (
                    f"#{int(records[i]['id'])} — "
                    f"{str(records[i].get('question') or '')[:60]}"
                ),
                key="dr_hist_pick",
            )
            if st.button("载入所选报告", key="dr_hist_load"):
                hid = int(records[pick_i]["id"])
                try:
                    row = get_research_report_by_id(hid)
                    if row and row.get("report_markdown"):
                        st.markdown(str(row["report_markdown"]))
                        if row.get("sources"):
                            with st.expander("引用行（research_report_sources）"):
                                st.dataframe(
                                    pd.DataFrame(row["sources"]),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    else:
                        st.warning("未找到该报告。")
                except Exception as e:
                    st.error(f"加载失败：{type(e).__name__}: {e}")

    # ================================================================
    # Tab 4 - 系统状态
    # ================================================================
    with tab4:
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown('<div class="section-header">🔑 API 与服务状态</div>', unsafe_allow_html=True)
            # LLM Key 状态
            if API_KEY and len(API_KEY) > 10:
                st.success("LLM API Key 已加载", icon="✅")
            else:
                st.error("LLM API Key 未配置（DASHSCOPE_API_KEY）", icon="❌")

            # Guardian Key 状态
            if GUARDIAN_API_KEY and len(GUARDIAN_API_KEY) > 5:
                st.success("Guardian API Key 已加载", icon="✅")
            else:
                st.warning("Guardian API Key 未配置（可选）", icon="⚠️")

            st.markdown("**数据库统计（看板数据源：MySQL）**")
            s1, s2, s3 = _cached_stats()
            st.caption(f"• article_extractions：{s1} 条")
            st.caption(f"• 去重标签（全库）：{s2} 个")
            st.caption(f"• 主域×子域组合：{s3} 种")
            st.caption(f"• 高频词池（展示 Top）：{kw_total} 个")
            st.caption(f"• MySQL：`{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}`")
            st.caption(f"• Agent 本地库（SQLite）：`{DB_PATH}`")

        with sc2:
            st.markdown('<div class="section-header">📡 信源配置</div>', unsafe_allow_html=True)
            st.caption("**卫报 Content API（已集成）**")
            st.caption("• 检索：AI safety / AI governance / AI regulation 等")
            st.caption("• 拉取字段：标题、导语、正文、版块、发布时间")
            st.caption("• 并发抽取：5 篇文章同时调用 LLM，串行入库")
            st.caption("**Crawl4AI（已集成，按 URL 侦察）**")
            st.caption("• 支持任意 URL：CSET、斯坦福 AI Index、OpenAI 博客等")
            st.caption("• 通过浏览器引擎渲染 JS 页面后提取结构化情报")

        st.divider()

        # --- 受密码保护的演示操作区 ---
        st.markdown('<div class="section-header">🔐 演示操作区（需验证）</div>', unsafe_allow_html=True)

        required_pwd = os.getenv("DEMO_PASSWORD", "").strip()
        if required_pwd:
            st.text_input(
                "演示密码",
                type="password",
                key="demo_pwd",
                placeholder="输入演示密码后解锁操作",
            )

        if _demo_unlocked():
            if not required_pwd:
                st.caption("（未设置 DEMO_PASSWORD 环境变量，操作区默认开放）")

            op1, op2 = st.columns(2)

            # ---- 卫报一键同步 ----
            with op1:
                st.markdown("**📡 卫报 AI 治理新闻同步**")
                sync_pages = st.slider("拉取页数", 1, 5, 2, key="sync_pages")
                sync_size = st.slider("每页条数", 3, 20, 8, key="sync_size")
                if st.button("🚀 一键同步卫报新闻", type="primary", use_container_width=True, key="btn_sync"):
                    with st.spinner("正在并发抽取，请稍候…"):
                        try:
                            from crawler.orchestrator import sync_guardian
                            r = sync_guardian(
                                max_pages=sync_pages,
                                page_size=sync_size,
                                rag_enabled=False,
                            )
                            st.cache_data.clear()
                            if r.saved > 0:
                                st.success(
                                    f"✅ 同步完成！入库 **{r.saved}** 条，"
                                    f"跳过已有 {r.skipped_url_dup} 条"
                                )
                            else:
                                st.info(
                                    f"同步完成：入库 {r.saved} 条，"
                                    f"跳过已有 {r.skipped_url_dup}，"
                                    f"无关 {r.skipped_no_incident}，失败 {r.failed}"
                                )
                            if r.new_keywords:
                                st.info(f"新增关键词：{', '.join(r.new_keywords[:8])}")
                            with st.expander("查看详细日志"):
                                for line in r.debug_log:
                                    st.caption(line)
                        except Exception as e:
                            st.error(f"同步失败：{type(e).__name__}: {e}")

            # ---- Agent URL 侦察 ----
            with op2:
                st.markdown("**🔍 Agent URL 深度侦察**")
                scout_presets = {
                    "CSET 新闻": "https://cset.georgetown.edu/news/",
                    "斯坦福 AI Index": "https://aiindex.stanford.edu/",
                    "OpenAI 博客": "https://openai.com/news/",
                    "EU AI Act": "https://artificialintelligenceact.eu/news/",
                }
                preset_sel = st.selectbox("预设信源", ["自定义"] + list(scout_presets.keys()), key="scout_preset")
                default_url = scout_presets.get(preset_sel, st.session_state.get("scout_url_val", ""))
                scout_url = st.text_input("目标 URL", value=default_url, key="scout_url_val")

                with st.expander("LLM 接口配置", expanded=False):
                    tab_api_key = st.text_input("API Key", value=API_KEY, type="password", key="scout_api_key")
                    tab_base_url = st.text_input("Base URL", value=BASE_URL, key="scout_base_url")

                if st.button("🕵️ 启动 Agent 侦察", type="primary", use_container_width=True, key="btn_scout"):
                    with st.spinner("Agent 正在深度分析中，请稍候…"):
                        try:
                            from crawler.agentic_crawl import run_agentic_crawl
                            incidents_data, new_keywords, debug_info = asyncio.run(
                                run_agentic_crawl(
                                    scout_url,
                                    api_key=tab_api_key or None,
                                    base_url=tab_base_url or None,
                                )
                            )
                        except Exception as e:
                            incidents_data, new_keywords, debug_info = [], [], [f"执行异常: {e}"]

                    with st.expander("调试日志"):
                        for line in debug_info:
                            st.caption(line)

                    if incidents_data:
                        saved_count = 0
                        for inc_dict in incidents_data:
                            try:
                                inc = incident_from_extraction(inc_dict)
                                ok, _ = save_incident(inc, scout_url)
                                if ok:
                                    saved_count += 1
                            except Exception:
                                pass
                        st.cache_data.clear()
                        st.success(
                            f"✅ 提取 **{len(incidents_data)}** 条情报，入库 **{saved_count}** 条"
                        )
                        if new_keywords:
                            st.info(f"新增关键词：{', '.join(new_keywords[:6])}")
                    else:
                        st.warning("未发现 AI 治理相关线索，或 URL 无法访问/LLM 未响应。")
        else:
            st.info("请输入正确的演示密码以解锁操作区。")

    # ================================================================
    # 侧边栏：仅展示项目简介（不放操作按钮）
    # ================================================================
    with st.sidebar:
        st.markdown("### 🛡️ 系统简介")
        st.markdown("""
**全球 AI 治理监测与自增长 Agent 系统**

自动感知全球 AI 安全动态，基于三元意图风险模型结构化分类，持续演化知识体系。

**核心能力**
- 卫报 Content API 定时同步
- 任意 URL 深度 Agent 侦察
- 问答式深度调研（混合检索 + 报告留痕）
- LLM 并发抽取（5 路并发）
- RAG 增强风险子域精炼
- 自增长关键词与子域体系

**技术栈**
Python · Streamlit · MySQL  
Crawl4AI · ChromaDB · httpx
        """)
        st.divider()
        st.caption(f"© {datetime.now().year} AI Safety Research")


if __name__ == "__main__":
    main()
