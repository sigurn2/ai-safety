"""
Streamlit 应用入口：AI 治理监测演示看板（汇报版）。

功能：三 Tab 只读看板（缓存全部查询，领导打开稳定不卡）；
     侧边栏受密码保护的操作区供现场演示触发同步与 Agent 侦察。
输入：SQLite 数据库（DB_PATH）；操作区依赖 LLM API Key 与 Guardian API Key。
输出：页面渲染；操作区副作用：网络请求 + SQLite 写入。
上下游：依赖 core.db、crawler.orchestrator、crawler.agentic_crawl；
        由 systemd 在服务器持续运行，Nginx 反代对外暴露。
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st

from core.config import API_KEY, BASE_URL, DB_PATH, GUARDIAN_API_KEY
from core.db import (
    get_risk_taxonomy_df,
    get_stats,
    get_watched_keywords,
    incident_from_extraction,
    init_db,
    save_incident,
)
from models.schema import RISK_DOMAIN_CHOICES

# Windows 下 Playwright 子进程兼容
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------------------------------------------------------------------------
# 缓存包装：所有只读查询加 2 分钟缓存，翻 Tab 不重查库
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def _cached_stats() -> Tuple[int, int, int]:
    """功能：缓存版 get_stats；输出：(incidents 数, 标签去重数, 子域种数)。"""
    try:
        return get_stats()
    except Exception:
        return 0, 0, 0


@st.cache_data(ttl=120)
def _cached_taxonomy() -> pd.DataFrame:
    """功能：缓存版 get_risk_taxonomy_df；输出：risk_taxonomy DataFrame。"""
    try:
        return get_risk_taxonomy_df()
    except Exception:
        return pd.DataFrame(columns=["domain", "subdomain", "count", "first_seen"])


@st.cache_data(ttl=120)
def _cached_keywords() -> pd.DataFrame:
    """功能：缓存版 get_watched_keywords；输出：watched_keywords DataFrame。"""
    try:
        return get_watched_keywords()
    except Exception:
        return pd.DataFrame(columns=["keyword", "count"])


@st.cache_data(ttl=60)
def _cached_latest_incidents(limit: int = 20) -> pd.DataFrame:
    """功能：缓存最新 incidents 列表；输入：limit；输出：DataFrame。"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT title,
                   COALESCE(NULLIF(trim(risk_level), ''), category) AS 风险等级,
                   risk_domain                                       AS 主域,
                   risk_subdomain                                    AS 子域,
                   entity                                            AS 涉及主体,
                   url                                               AS 来源,
                   timestamp                                         AS 时间
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _cached_all_incidents() -> pd.DataFrame:
    """功能：缓存全量 incidents 供详情 Tab 筛选；输出：DataFrame。"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT id,
                   title                                             AS 标题,
                   COALESCE(NULLIF(trim(risk_level), ''), category) AS 风险等级,
                   risk_domain                                       AS 主域,
                   risk_subdomain                                    AS 子域,
                   entity                                            AS 涉及主体,
                   content                                           AS 摘要,
                   url                                               AS 来源,
                   tags                                              AS 标签,
                   timestamp                                         AS 时间
            FROM incidents
            ORDER BY timestamp DESC
            """,
            conn,
        )
        conn.close()
        return df
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
    输出：无；副作用：init_db；操作区触发时写 SQLite 与网络请求。
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

    # --- 三 Tab 看板 ---
    tab1, tab2, tab3 = st.tabs(["📊 监测看板", "📋 情报详情", "⚙️ 系统状态"])

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
            st.markdown('<div class="section-header" style="margin-top:24px">🌳 动态风险分类体系（三元主域 → 子域）</div>', unsafe_allow_html=True)
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
                                st.caption(f"· {row['subdomain']}（×{int(row['count'])}）")
            else:
                st.caption("子域数据积累中，入库带 risk_subdomain 的情报后自动更新。")

        with right:
            # 风险主域分布饼图（用 bar chart 替代，无需额外库）
            st.markdown('<div class="section-header">📊 风险主域分布</div>', unsafe_allow_html=True)
            tax_df_r = _cached_taxonomy()
            if not tax_df_r.empty:
                # 按主域聚合 count
                domain_agg = (
                    tax_df_r.groupby("domain")["count"].sum().reset_index()
                )
                domain_agg["主域"] = (
                    domain_agg["domain"].str.replace(r"\s*\(.+$", "", regex=True).str.strip()
                )
                domain_agg = domain_agg.rename(columns={"count": "情报数"})
                st.bar_chart(domain_agg.set_index("主域")["情报数"])

                # Top 子域频次
                st.markdown('<div class="section-header" style="margin-top:20px">🔥 高频风险子域 Top 12</div>', unsafe_allow_html=True)
                top_sub = tax_df_r.sort_values("count", ascending=False).head(12).copy()
                short_dom = top_sub["domain"].str.replace(r"\s*\(.+$", "", regex=True)
                top_sub["子域"] = top_sub["subdomain"] + " · " + short_dom
                st.bar_chart(top_sub.set_index("子域")["count"].rename("次数"))
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
                domains = ["全部"] + sorted(df_all["主域"].dropna().unique().tolist())
                sel_domain = st.selectbox("按主域筛选", domains, key="filter_domain")
            with fc2:
                levels = ["全部"] + sorted(df_all["风险等级"].dropna().unique().tolist())
                sel_level = st.selectbox("按风险等级筛选", levels, key="filter_level")
            with fc3:
                kw_search = st.text_input("关键词搜索（标题/摘要）", key="kw_search")

            df_view = df_all.copy()
            if sel_domain != "全部":
                df_view = df_view[df_view["主域"] == sel_domain]
            if sel_level != "全部":
                df_view = df_view[df_view["风险等级"] == sel_level]
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
                report_md += "### 一、核心风险预警\n\n"
                for _, row in df_report.iterrows():
                    lvl = str(row.get("风险等级", "") or "").strip()
                    dom = str(row.get("主域", "") or "").strip()
                    sub = str(row.get("子域", "") or "").strip()
                    entity = str(row.get("涉及主体", "") or "").strip()
                    tri = f"{dom} / {sub}".strip(" /")
                    report_md += f"- **[{lvl or '—'}]** {row['title']}（涉及主体：{entity or '—'}）"
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
    # Tab 3 - 系统状态
    # ================================================================
    with tab3:
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

            st.markdown("**数据库统计**")
            s1, s2, s3 = _cached_stats()
            st.caption(f"• incidents 表：{s1} 条")
            st.caption(f"• risk_taxonomy：{taxonomy_kinds} 个子域")
            st.caption(f"• watched_keywords：{kw_total} 个词")
            st.caption(f"• 数据库路径：`{DB_PATH}`")

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
- LLM 并发抽取（5 路并发）
- RAG 增强风险子域精炼
- 自增长关键词与子域体系

**技术栈**
Python · Streamlit · SQLite  
Crawl4AI · ChromaDB · httpx
        """)
        st.divider()
        st.caption(f"© {datetime.now().year} AI Safety Research")


if __name__ == "__main__":
    main()
