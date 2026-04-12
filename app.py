import streamlit as st
import asyncio
import sys  
# 解决 Windows 下 Playwright 子进程报错 
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import hashlib
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# 优先加载 .env 文件，覆盖系统环境变量
load_dotenv(override=True)

# --- 全局配置常量（统一从 .env 读取）---
API_KEY   = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# --- 1. 核心导入兼容性处理 ---
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from typing import Optional, List, Any, Tuple  # 类型提示

try:
    from crawl4ai.async_configs import LLMConfig # type: ignore
except:
    from crawl4ai.config import LLMConfig # type: ignore

from schema import AIIncident, ExtractionResult, RISK_DOMAIN_CHOICES  # type: ignore

# --- 2. 数据库操作逻辑 ---
DB_PATH = 'ai_governance.db'


def _table_column_names(conn: sqlite3.Connection, table: str) -> List[str]:
    """读取 SQLite 表当前有哪些列（用于兼容旧版 incidents 表，避免升级后无法启动）。"""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def _migrate_incidents_schema(conn: sqlite3.Connection) -> None:
    """
    在保留旧数据的前提下，为 incidents 追加三元模型相关列。
    历史版本中「风险等级」同步拷到 risk_level。
    """
    cols = set(_table_column_names(conn, "incidents"))
    if "risk_level" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_level TEXT")
    if "risk_domain" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_domain TEXT")
    if "risk_subdomain" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_subdomain TEXT")
    if "category" in cols:
        conn.execute(
            """
            UPDATE incidents
            SET risk_level = category
            WHERE (risk_level IS NULL OR trim(risk_level) = '')
              AND category IS NOT NULL AND trim(category) != ''
            """
        )

#初始化数据库
def init_db():
    """初始化/迁移本地 SQLite：事件表、关键词池、风险分类演进表。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 先创建与历史版本一致的 incidents 骨架；新列一律通过迁移追加，老库也能平滑升级
    c.execute(
        """CREATE TABLE IF NOT EXISTS incidents
                 (id TEXT PRIMARY KEY, #事件id
                 title TEXT, #事件标题
                 category TEXT, #事件分类
                 entity TEXT, #事件涉及主体
                 content TEXT, #事件内容
                 url TEXT, #事件url
                 tags TEXT, #事件标签
                 timestamp DATETIME  #事件时间)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS watched_keywords
                 (keyword TEXT PRIMARY KEY, #关键词
                 first_seen DATETIME, #首次出现时间
                 count INTEGER DEFAULT 1 #出现次数)"""
    )
    # 动态子域演进表：主域 + 子域 联合主键，统计出现次数（自增长分类体系的核心存储）
    c.execute(
        """CREATE TABLE IF NOT EXISTS risk_taxonomy
                 (domain TEXT NOT NULL,
                  subdomain TEXT NOT NULL,
                  first_seen DATETIME,
                  count INTEGER DEFAULT 1,
                  PRIMARY KEY (domain, subdomain))"""
    )
    _migrate_incidents_schema(conn) #迁移 incidents 表
    conn.commit()
    conn.close()


def coerce_risk_domain(raw: Optional[str]) -> str:
    """
    将 LLM 可能返回的近似表述归一到三条 canonical 主域字符串之一。
    避免模型少写一个括号就导致整条解析失败。
    """
    if not raw or not isinstance(raw, str):
        return RISK_DOMAIN_CHOICES[2]
    v = raw.strip()
    if v in RISK_DOMAIN_CHOICES:
        return v
    low = v.lower()
    if "恶意" in v or "malicious" in low or "abuse" in low or "攻击" in v:
        return RISK_DOMAIN_CHOICES[0]
    if "意外" in v or "失效" in v or "鲁棒" in v or "accidental" in low or "failure" in low or "halluc" in low:
        return RISK_DOMAIN_CHOICES[1]
    if "系统" in v or "伦理" in v or "systemic" in low or "ethical" in low or "bias" in low or "偏见" in v:
        return RISK_DOMAIN_CHOICES[2]
    return RISK_DOMAIN_CHOICES[2]


def incident_from_extraction(d: dict) -> AIIncident:
    """把 LLM 返回的单条 dict 清洗后转为 AIIncident（补全缺省主域/子域）。"""
    data = dict(d)
    data["risk_domain"] = coerce_risk_domain(data.get("risk_domain"))
    sub = data.get("risk_subdomain")
    if sub is None or str(sub).strip() == "":
        data["risk_subdomain"] = "未指定子域"
    else:
        data["risk_subdomain"] = str(sub).strip()[:160]
    return AIIncident(**data)


def _bump_risk_taxonomy_cursor(c: sqlite3.Cursor, domain: str, subdomain: str) -> bool:
    """
    在同一事务内更新 risk_taxonomy：已存在则 count+1，否则插入。
    返回 True 表示本次为「全新 (主域, 子域) 组合」。
    """
    domain = (domain or "").strip()
    subdomain = (subdomain or "").strip()
    if not domain or not subdomain:
        return False
    c.execute(
        "SELECT count FROM risk_taxonomy WHERE domain = ? AND subdomain = ?",
        (domain, subdomain),
    )
    row = c.fetchone()
    if row:
        c.execute(
            "UPDATE risk_taxonomy SET count = count + 1 WHERE domain = ? AND subdomain = ?",
            (domain, subdomain),
        )
        return False
    c.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, ?, 1)",
        (domain, subdomain, datetime.now()),
    )
    return True


def get_stats() -> Tuple[int, int, int]:
    """返回 (事件条数, 不重复标签数, 风险子域种类数)。"""
    conn = sqlite3.connect(DB_PATH)
    count = int(pd.read_sql_query("SELECT COUNT(*) as total FROM incidents", conn).iloc[0]["total"])
    tags = pd.read_sql_query("SELECT tags FROM incidents", conn)
    try:
        tax_n = int(pd.read_sql_query("SELECT COUNT(*) AS n FROM risk_taxonomy", conn).iloc[0]["n"])
    except Exception:
        tax_n = 0
    conn.close()
    unique_tags: set = set()
    if not tags.empty and "tags" in tags.columns:
        for sublist in tags["tags"].dropna().astype(str).str.split(","):
            if sublist is not None:
                unique_tags.update(t for t in sublist if t)
    return count, len(unique_tags), tax_n


def get_risk_taxonomy_df() -> pd.DataFrame:
    """读取完整风险分类演进表，供看板与图表使用。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT domain, subdomain, count, first_seen FROM risk_taxonomy ORDER BY domain, count DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame(columns=["domain", "subdomain", "count", "first_seen"])
    conn.close()
    return df


def save_incident(incident: AIIncident, source_url: str = "") -> Tuple[bool, bool]:
    """
    保存一条事件到 incidents，并在同一事务内更新 risk_taxonomy。
    返回 (是否成功插入 incidents, 是否首次出现该 主域+子域 组合)。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    inc_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(incident.title.encode()).hexdigest()[:6]}"
    tag_str = ",".join(incident.tags)
    now = datetime.now()
    try:
        c.execute(
            """INSERT INTO incidents
               (id, title, risk_level, risk_domain, risk_subdomain, entity, content, url, tags, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                inc_id,
                incident.title,
                incident.risk_level,
                incident.risk_domain,
                incident.risk_subdomain,
                incident.entity,
                incident.summary,
                source_url,
                tag_str,
                now,
            ),
        )
        tax_new = _bump_risk_taxonomy_cursor(c, incident.risk_domain, incident.risk_subdomain)
        conn.commit()
        return True, tax_new
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, False
    finally:
        conn.close()


def get_watched_keywords() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT keyword, count FROM watched_keywords ORDER BY count DESC LIMIT 60", conn
        )
    except Exception:
        df = pd.DataFrame(columns=['keyword', 'count'])
    conn.close()
    return df


def update_watched_keywords(new_tags: list) -> list:
    """将新标签加入待观察关键词池，返回本次全新入库的标签列表（自增长核心）"""
    if not new_tags:
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        existing_df = pd.read_sql_query("SELECT keyword FROM watched_keywords", conn)
        existing = set(existing_df['keyword'].str.strip().tolist())
    except Exception:
        existing = set()

    newly_added = []
    for tag in new_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag in existing:
            c.execute("UPDATE watched_keywords SET count = count + 1 WHERE keyword = ?", (tag,))
        else:
            c.execute("INSERT OR IGNORE INTO watched_keywords VALUES (?, ?, 1)", (tag, datetime.now()))
            newly_added.append(tag)
            existing.add(tag)
    conn.commit()
    conn.close()
    return newly_added


# --- 3. 核心爬虫与 Agent 逻辑 ---
async def run_agentic_crawl(url: str, api_key: Optional[str] = None, base_url: Optional[str] = None, debug: bool = False):
    """
    执行 Agent 侦察任务：
    - 抓取目标 URL 并语义提取 AI 治理相关事件（含三元风险主域 + 动态子域）
    - 自动更新待观察关键词池；子域演进在入库时写入 risk_taxonomy 表
    返回: (incidents_list, newly_added_keywords, debug_info)
    """
    # IS_DEMO_MODE = False  # 网络不稳定时改为 True 
    debug_log = []

    # if IS_DEMO_MODE:
    #     await asyncio.sleep(1.5)
    #     demo_incidents = [{
    #         "title": "CSET: 2026年全球AI治理关键节点预测",
    #         "entity": "Georgetown CSET",
    #         "risk_level": "中",
    #         "summary": "报告指出本年度主要经济体将在大模型合规性上达成初步框架，重点关注 Agent 自主权安全审计。",
    #         "tags": ["Agent Autonomy", "Safety Audit", "Model Compliance", "EU AI Act", "AI Governance"]
    #     }]
    #     new_kw = update_watched_keywords(
    #         ["Agent Autonomy", "Safety Audit", "Model Compliance", "EU AI Act", "AI Governance"]
    #     )
    #     return demo_incidents, new_kw, ["[演示模式] 返回示例数据"]

    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    _api_key  = api_key  or API_KEY
    _base_url = base_url or BASE_URL
    _model = f"openai/{LLM_MODEL}"  #
    # 运行级可配置超时：默认 90s，降低跨境站点偶发慢加载导致的 30s 导航超时
    try:
        crawl_page_timeout_ms = int(os.getenv("CRAWL_PAGE_TIMEOUT_MS", "90000"))
    except Exception:
        crawl_page_timeout_ms = 90000
    # Playwright page.goto 的 wait_until：默认 commit（比 domcontentloaded 更早放行），避免首包已回但 DCL 长期不触发导致超时
    crawl_wait_until = (os.getenv("CRAWL_WAIT_UNTIL", "commit") or "commit").strip()
    # 使用运行级 session_id，避免固定会话在命中反爬后“粘住”后续请求
    crawl_session_id = f"ai_monitor_{run_id}"
    if not _api_key:
        debug_log.append("❌ API Key 未配置，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        return [], [], debug_log

    try:
        debug_log.append(f"✓ API 配置已验证（模型: {_model}）")

        llm_config = LLMConfig(
            provider=_model,
            api_token=_api_key,
            base_url=_base_url
        )
        #核心提取策略
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=ExtractionResult.model_json_schema(),
            #extra_args=extra_args,
            instruction=(
                "你是一个 AI 治理与安全领域的专家分析师。仔细分析网页内容，识别所有与 AI 治理、AI 安全、AI 政策、AI 监管相关的内容。\n"
                "对每条相关内容，精确提取以下字段（JSON 格式）：\n"
                "1. title（标题）：事件、报告、新闻或会议的标题\n"
                "2. entity（涉及主体）：提及的机构、公司、政府或人物名称\n"
                "3. risk_level（风险等级）：根据内容判断，填写「高」「中」「低」之一\n"
                "4. risk_domain（风险主域 — 意图与来源三元模型）：必须从以下三项中**原样**选一条字符串（含英文与中文括号）：\n"
                "   - Malicious Use (恶意滥用)：人类恶意利用 AI，或对 AI 系统发起主动攻击（越狱、投毒、深度伪造诈骗、自动化网络攻击等）。\n"
                "   - Accidental Failure (意外失效)：无恶意攻击者，因系统缺陷、幻觉或复杂环境下的失效（严重幻觉、自动驾驶误判、域外泛化失败等）。\n"
                "   - Systemic & Ethical Risk (系统性与伦理风险)：系统按预期运行，但对社会或个人权益产生负面影响（算法偏见、隐私、版权、信息茧房、就业冲击等）。\n"
                "5. risk_subdomain（风险子域）：在该主域下的简短专业子类型（如「越狱攻击」「数据投毒」「算法偏见」）。同类现象用语尽量一致，新现象可创造清晰新词。\n"
                "6. summary（摘要）：一句话不超过 60 字\n"
                "7. tags（标签）：3-8 个关键词，中英文均可\n\n"
                "重要提示：\n"
                "- 只提取与 AI 治理/安全相关的内容；risk_domain 三条字符串必须与上文完全一致\n"
                "- 不要捏造事实\n"
                "- 若无相关内容，incidents 为空数组\n"
                "- 返回 JSON：{\"incidents\": [...]} "
            )
        )

        # 不在此设置 wait_for：AsyncPlaywright 策略在 goto 后已 wait_for_selector("body")，
        # 再设 wait_for="body" 会进入第二轮 smart_wait，在反爬/挑战页上易重复超时。
        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy, #核心提取策略
            cache_mode=CacheMode.BYPASS, #不使用缓存，每次都重新爬取网页
            wait_until=crawl_wait_until,  # 与 Playwright page.goto 一致；见 crawl4ai async_crawler_strategy goto
            page_timeout=crawl_page_timeout_ms,  # 可配置超时：默认 90 秒
            session_id=crawl_session_id,  # 运行级 session，减少反爬状态跨次污染
            js_code="window.scrollTo(0, document.body.scrollHeight);" # 模拟滚动以触发懒加载
        )

        async with AsyncWebCrawler() as crawler:
            debug_log.append("📡 正在爬取目标 URL...")
            # 这里的 result 其实就是普通的 object
            result: Any = await crawler.arun(url=url, config=config) # 爬虫返回结果

            debug_log.append(f"✓ 爬虫返回: success={result.success}")

            if not result.success:
                debug_log.append(f"❌ 爬虫失败: {result.error_message if hasattr(result, 'error_message') else '未知错误'}")
                return [], [], debug_log

            if not result.extracted_content:
                debug_log.append("❌ 爬虫获取内容为空，页面可能被屏蔽或不存在")
                return [], [], debug_log

            debug_log.append(f"✓ 获取到内容（长度: {len(str(result.extracted_content))} 字符）")

            try:
                raw = result.extracted_content # 原始内容
                data = json.loads(raw) if isinstance(raw, str) else raw   
                debug_log.append(f"✓ LLM 提取完成，返回类型: {type(data)}")
            except Exception as e:
                debug_log.append(f"❌ JSON 解析失败: {str(e)}")
                debug_log.append(f"   原始内容: {str(raw)[:200]}")
                return [], [], debug_log

            # --- 兼容性修复逻辑 ---
            incidents_data = []
            if isinstance(data, list):
                # 如果 LLM 直接返回了列表，直接使用
                incidents_data = data
                debug_log.append("💡 LLM 直接返回了 List 格式")
            elif isinstance(data, dict):
                # 如果返回的是字典，尝试获取 incidents 键，如果没有则把字典转为列表
                incidents_data = data.get("incidents", [])
                if not incidents_data and data:
                    # 防止 LLM 把单条数据当成 dict 根节点返回
                    incidents_data = [data]
                debug_log.append("💡 LLM 返回了 Dict 格式")
            else:
                debug_log.append(f"❌ LLM 返回格式错误：期望 dict 或 list，实际 {type(data)}")
                return [], [], debug_log

            debug_log.append(f"✓ 最终提取到 {len(incidents_data)} 条情报")

            if not incidents_data:
                debug_log.append("💡 可能原因：1) 页面内容无相关信息 2) Schema 验证失败 3) LLM 输出格式异常")
                return [], [], debug_log

            # 自增长逻辑：收集本次所有标签，与关键词池比对
            all_tags = []
            for inc in incidents_data:
                all_tags.extend(inc.get("tags", []))
            newly_added = update_watched_keywords(all_tags)
            debug_log.append(f"📊 新增关键词: {len(newly_added)}")

            return incidents_data, newly_added, debug_log

    except Exception as e:
        debug_log.append(f"❌ 执行异常: {type(e).__name__}: {str(e)}")
        import traceback
        debug_log.append(f"   堆栈: {traceback.format_exc()[:300]}")
        return [], [], debug_log


# --- 4. UI 界面与交互 ---
def main():
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
