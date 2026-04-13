"""
Streamlit 应用入口：监测看板与爬虫调度 UI。

功能：展示指标、触发异步侦察、将结果写入本地库；抽取/RAG/关键词池等业务细节在 crawler 与 engine 中完成。
输入：用户操作与环境变量（由 core.config 加载）；数据库路径 DB_PATH。
输出：页面渲染；副作用：SQLite 读写、网络请求（经 run_agentic_crawl）。
上下游：调用 crawler.agentic_crawl；持久化经 core.db；领域模型见 models.schema。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st

from core.config import API_KEY, BASE_URL, DB_PATH
from core.db import (
    get_risk_taxonomy_df,
    get_stats,
    get_watched_keywords,
    incident_from_extraction,
    init_db,
    save_incident,
)
from crawler.agentic_crawl import run_agentic_crawl
from models.schema import RISK_DOMAIN_CHOICES

# Windows 下 Playwright 子进程
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# --- UI 界面与交互 ---


def main() -> None:
    """
    功能：配置页面、拉取统计、渲染侧边栏侦察与多 Tab 看板。
    输入：无参数；依赖 Streamlit session 与全局已加载的 core.config。
    输出：无；副作用：init_db、可能触发 asyncio.run(run_agentic_crawl)。
    """
    st.set_page_config(page_title="MIIT AI Governance Monitoring", layout="wide", page_icon="🛡️")
    init_db()

    st.markdown("""
        <style>
        .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00f2fe; }
        .report-box { background-color: #f0f2f6; color: #1e2130; padding: 20px; border-radius: 10px; font-family: 'Courier New'; }
        .tag-chip {
            background: #1e2130; color: #00f2fe; padding: 4px 12px;
            border-radius: 15px; margin: 3px; border: 1px solid #00f2fe;
            display: inline-block; font-size: 13px;
        }
        .scout-placeholder {
            background: #1e2130; padding: 50px; border-radius: 12px;
            text-align: center; border: 2px dashed #2a3050; margin-top: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ 全球 AI 治理监测与自增长 Agent 系统")

    # 动态获取指标（第三项为 risk_taxonomy 中不同「主域+子域」组合数量）
    total_incidents, total_tags, taxonomy_kinds = get_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("已监测信源", "142", "+5")
    m2.metric("识别风险线索", total_incidents, f"+{total_incidents}")
    m3.metric("知识库节点", "856", "稳定")
    m4.metric("自增长标签 / 子域种数", f"{total_tags} / {taxonomy_kinds}", help="标签去重数与动态风险子域种类数")

    # 侧边栏
    with st.sidebar:
        st.header("🤖 Agent 调度中心")

        # .env 加载状态指示
        if API_KEY and API_KEY != "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            st.success("🔑 .env 密钥已加载", icon="✅")
        else:
            st.warning("⚠️ 请在 .env 文件中填写 DASHSCOPE_API_KEY", icon="⚠️")

        target_url = st.text_input("监测目标 URL", "https://cset.georgetown.edu/news/")

        with st.expander("⚙️ LLM API 配置"):
            sidebar_api_key = st.text_input(
                "API Key", value=API_KEY,
                type="password", key="sidebar_api_key"
            )
            sidebar_base_url = st.text_input(
                "Base URL", value=BASE_URL,
                key="sidebar_base_url"
            )

        if st.button("🚀 启动深度感知回路"):
            with st.status("Agent 正在进化...", expanded=True) as status:
                try:
                    incidents_data, new_keywords, debug_info = asyncio.run(
                        run_agentic_crawl(
                            target_url,
                            api_key=sidebar_api_key or None,
                            base_url=sidebar_base_url or None
                        )
                    )

                    # 显示调试日志
                    for log_line in debug_info:
                        st.write(log_line)

                    if incidents_data:
                        new_sub_pairs: List[str] = []
                        for inc_dict in incidents_data:
                            try:
                                inc = incident_from_extraction(inc_dict)
                                ok, tax_new = save_incident(inc, target_url)
                                if ok:
                                    st.write(
                                        f"✅ 发现新线索: **{inc.title}** "
                                        f"〔{inc.risk_domain.split('(')[0].strip()} / {inc.risk_subdomain}〕"
                                    )
                                    if tax_new:
                                        new_sub_pairs.append(f"{inc.risk_subdomain}")
                            except Exception:
                                pass
                        if new_keywords:
                            st.write(f"🧬 新增进化标签: {', '.join(new_keywords)}")
                        if new_sub_pairs:
                            st.write(f"🌳 本次新出现的风险子域: {', '.join(new_sub_pairs)}")
                        status.update(label="感知完成！知识库已更新", state="complete")
                    else:
                        status.update(label="未发现新线索或提取失败", state="error")
                except Exception as e:
                    st.error(f"调度失败: {e}")
            st.balloons()

    # 主界面 Tab 分区
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 治理监测看板",
        "🔗 数据血缘与溯源",
        "📝 自动化汇报生成",
        "🔍 深度感知回路"
    ])

    with tab1:
        c_l, c_r = st.columns([2, 1])
        with c_l:
            st.subheader("📍 最新监测动态")
            conn = sqlite3.connect(DB_PATH)
            # COALESCE：兼容仅存在旧列 category、尚未回填 risk_level 的历史行
            df = pd.read_sql_query(
                """
                SELECT title,
                       COALESCE(NULLIF(trim(risk_level), ''), category) AS risk_level,
                       risk_domain,
                       risk_subdomain,
                       entity,
                       timestamp
                FROM incidents
                ORDER BY timestamp DESC
                LIMIT 10
                """,
                conn,
            )
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无监测数据，请从侧边栏启动感知。")
            conn.close()

            # 三元模型 + 动态子域：只读展示 risk_taxonomy（关系库存储，便于日后同步到图数据库）
            st.subheader("🌳 动态风险分类体系（三元主域 → 子域）")
            tax_df = get_risk_taxonomy_df()
            if tax_df.empty:
                st.caption("尚无子域数据；成功入库带 risk_subdomain 的事件后，此处会自动累积。")
            else:
                dom_cols = st.columns(3)
                for i, domain_label in enumerate(RISK_DOMAIN_CHOICES):
                    short = domain_label.split("(")[0].strip()
                    sub_df = tax_df[tax_df["domain"] == domain_label].head(12)
                    with dom_cols[i]:
                        st.markdown(f"**{short}**")
                        if sub_df.empty:
                            st.caption("—")
                        else:
                            for _, row in sub_df.iterrows():
                                st.caption(f"- {row['subdomain']}（×{int(row['count'])}）")

        with c_r:
            st.subheader("🔥 子域出现频次（Top）")
            tax_df_chart = get_risk_taxonomy_df()
            if tax_df_chart.empty:
                st.caption("暂无统计数据。")
            else:
                topn = tax_df_chart.sort_values("count", ascending=False).head(12).copy()
                # 横轴标签带上主域简写，避免不同主域下出现同名子域时难以区分
                short_dom = topn["domain"].astype(str).str.replace(r"\s*\(.+$", "", regex=True)
                topn["label"] = topn["subdomain"].astype(str) + " · " + short_dom
                chart_data = pd.DataFrame(
                    {"子域": topn["label"], "次数": topn["count"].astype(int)}
                )
                st.bar_chart(chart_data, x="子域", y="次数", color="#00f2fe")

    with tab2:
        st.subheader("🕸️ 自动化自增长血缘图")
        st.image("https://mermaid.ink/svg/pako:eNqNkk9v2zAMxb8KwfS0B_8dChTYofSwaYF1K-ZhuSlyYsuRJUuOnG7I996T7SRp0S09mSJFv_fI90iK9mY1m6Nl0pX_OarO1V0f_Z_v-rNf6R_f6f_f_v9t_9-m_z_0_6v-v_X_vS_775_7_0_9_6X_f_r_rf-37X_v-v_X_p_v-v_7_r9P_3_of-P7X_p-Z_uf_f-u_0_7_6X_f-v_X_v-V_9Xvve_9H_l-_9L_-_U_5f-f-n_nf9v-3_T_7f9_6n_X_r-V7-P_f_p_3_-n-v_f_n_p_9_9P9X_4_9f_f_R__f9P_h_0v_X9X-v-z_W_-v_X-u_0_7v-v_X_r-V79Pvb_p_0v-X_n_p_5f-f-n_f")

    with tab3:
        st.subheader("📄 课题成果自动导出")
        if st.button("📥 一键生成 AI 治理监测日报"):
            conn = sqlite3.connect(DB_PATH)
            today_data = pd.read_sql_query(
                "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 5", conn
            )
            conn.close()

            if not today_data.empty:
                report_md = f"## 📅 AI 治理动态监测内参 ({datetime.now().strftime('%Y-%m-%d')})\n\n"
                report_md += "### 一、 核心风险预警\n"
                for _, row in today_data.iterrows():
                    lvl = row.get("risk_level")
                    if lvl is None or (isinstance(lvl, float) and pd.isna(lvl)) or str(lvl).strip() == "":
                        lvl = row.get("category", "")
                    dom = row.get("risk_domain", "") or ""
                    sub = row.get("risk_subdomain", "") or ""
                    tri = f"{dom} / {sub}".strip(" /")
                    report_md += f"- **[{lvl}]** {row['title']} (涉及主体: {row['entity']})"
                    if tri:
                        report_md += f" — 分类: {tri}"
                    report_md += "\n"

                report_md += "\n### 二、 新兴术语与概念感知\n"
                all_tags = ",".join(today_data['tags'].fillna('')).split(',')
                report_md += f"- **本期新词：** {', '.join(list(set(all_tags))[:5])}\n"

                st.markdown(f'<div class="report-box">{report_md}</div>', unsafe_allow_html=True)
                st.download_button("下载 Markdown 报告", report_md, file_name="AI_Governance_Daily.md")
            else:
                st.warning("数据库中尚无足够数据生成日报。")

    # ===== Tab 4: 深度感知回路 =====
    with tab4:
        st.subheader("🔍 深度感知回路 — Agent 自主侦察")
        st.caption("输入任意目标 URL，Agent 将自动提取 AI 治理相关线索，并进化知识库关键词池")

        ctrl_col, result_col = st.columns([1, 2], gap="large")

        with ctrl_col:
            st.markdown("**🎯 侦察目标**")
            # 从预设按钮传入的 URL 用独立 key，避免与 widget key 冲突
            _default_url = st.session_state.pop("_preset_url", "https://cset.georgetown.edu/news/")
            scout_url = st.text_input(
                "目标 URL",
                value=_default_url,
                label_visibility="collapsed",
                placeholder="输入待分析的网页地址..."
            )

            with st.expander("⚙️ LLM 接口配置", expanded=False):
                tab_api_key = st.text_input(
                    "API Key", value=API_KEY,
                    type="password", key="tab_api_key",
                    placeholder="sk-..."
                )
                tab_base_url = st.text_input(
                    "Base URL", value=BASE_URL,
                    key="tab_base_url"
                )

            launch_btn = st.button(
                "🕵️ 启动 Agent 侦察", type="primary", use_container_width=True
            )

            if launch_btn:
                with st.spinner("🧠 Agent 正在深度分析中，请稍候..."):
                    incidents_data, new_keywords, debug_info = asyncio.run(
                        run_agentic_crawl(
                            scout_url,
                            api_key=tab_api_key or None,
                            base_url=tab_base_url or None
                        )
                    )

                # 显示调试日志
                with st.expander("📋 调试日志", expanded=False):
                    for log_line in debug_info:
                        st.caption(log_line)

                if incidents_data:
                    # 入库（与侧边栏共用：三元主域 + 子域写入 incidents，并 bump risk_taxonomy）
                    saved_count = 0
                    new_subs_session: List[str] = []
                    for inc_dict in incidents_data:
                        try:
                            inc = incident_from_extraction(inc_dict)
                            ok, tax_new = save_incident(inc, scout_url)
                            if ok:
                                saved_count += 1
                                if tax_new:
                                    new_subs_session.append(inc.risk_subdomain)
                        except Exception:
                            pass

                    # 更新 session state
                    st.session_state["scout_results"] = incidents_data
                    st.session_state["scout_new_kw"] = new_keywords
                    st.session_state["scout_new_subdomains"] = new_subs_session
                    st.session_state["scout_url_used"] = scout_url

                    st.success(
                        f"✅ 发现新线索，系统已自动进化！"
                        f"共提取 **{len(incidents_data)}** 条情报，入库 **{saved_count}** 条。"
                    )
                    if new_keywords:
                        preview = ', '.join(new_keywords[:5])
                        extra = f" …等 {len(new_keywords)} 个" if len(new_keywords) > 5 else f"，共 {len(new_keywords)} 个"
                        st.info(f"🧬 新增进化标签：{preview}{extra}")
                    if new_subs_session:
                        st.info(f"🌳 本次新收录的风险子域：{', '.join(new_subs_session)}")
                else:
                    st.warning(
                        "⚠️ 未发现 AI 治理相关线索。\n\n"
                        "**可能原因：**\n"
                        "1. 页面无相关内容（不是 AI 治理话题）\n"
                        "2. 爬虫被页面屏蔽\n"
                        "3. LLM API 配置有误\n\n"
                        "**建议：** 查看上方「调试日志」或尝试其他 URL"
                    )

        with result_col:
            st.markdown("**📊 实时提取结果**")
            if st.session_state.get("scout_results"):
                url_used = st.session_state.get("scout_url_used", "")
                st.caption(f"来源: `{url_used}`")
                st.json(st.session_state["scout_results"])
            else:
                st.markdown("""
                <div class="scout-placeholder">
                    <p style="color:#556;font-size:20px;margin:0;">📡 等待侦察指令...</p>
                    <p style="color:#445;font-size:13px;margin-top:10px;">
                        在左侧输入目标 URL，点击「启动 Agent 侦察」后，<br>
                        结构化情报将在此实时展示
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # 待观察关键词池
        st.divider()

        # ---- 推荐爬虫网址 ----
        st.subheader("💡 推荐爬虫网址")
        st.caption("点击快速填入 URL（含丰富的 AI 治理内容）")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        test_urls = {
            "🔷 CSET 新闻":     ("preset_cset",    "https://cset.georgetown.edu/news/"),
            "🔶 AI Index":       ("preset_aiindex", "https://aiindex.stanford.edu/"),
            "🔴 OpenAI 博客":   ("preset_openai",  "https://openai.com/news/"),
        }
        for idx, (label, (btn_key, url_val)) in enumerate(test_urls.items()):
            col = [preset_col1, preset_col2, preset_col3][idx]
            with col:
                if st.button(label, use_container_width=True, key=btn_key):
                    st.session_state["_preset_url"] = url_val
                    st.rerun()

        st.divider()
        pool_col, stat_col = st.columns([3, 1])

        with pool_col:
            st.subheader("🧬 待观察关键词池（自增长）")
            kw_df = get_watched_keywords()
            if not kw_df.empty:
                tag_html = "".join([
                    f'<span class="tag-chip">'
                    f'{row["keyword"]} '
                    f'<span style="opacity:0.5;font-size:11px;">×{row["count"]}</span>'
                    f'</span>'
                    for _, row in kw_df.iterrows()
                ])
                st.markdown(tag_html, unsafe_allow_html=True)
            else:
                st.caption("🌱 关键词池为空，启动一次侦察以填充初始数据。")

        with stat_col:
            st.subheader("📈 进化统计")
            kw_total = len(kw_df) if not kw_df.empty else 0
            st.metric("关键词总量", kw_total, help="已观察的 AI 治理领域关键词总数")
            new_session_kw = len(st.session_state.get("scout_new_kw", []))
            st.metric("本次新增", new_session_kw, help="本次侦察新发现的关键词数量")
            new_sub_n = len(st.session_state.get("scout_new_subdomains", []))
            st.metric("本次新子域", new_sub_n, help="本次侦察首次出现的 主域+子域 组合数")


if __name__ == "__main__":
    main()
