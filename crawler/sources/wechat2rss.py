from typing import Optional, List
from pydantic import BaseModel, Field
import feedparser
import requests
from bs4 import BeautifulSoup

pool = {
    # "量子位":"https://wechat2rss.xlab.app/feed/7131b577c61365cb47e81000738c10d872685908.xml",
    "新智元":"https://wechat2rss.xlab.app/feed/ede30346413ea70dbef5d485ea5cbb95cca446e7.xml",
    "机器之心":"https://wechat2rss.xlab.app/feed/51e92aad2728acdd1fda7314be32b16639353001.xml",
    "中国信息安全":"https://wechat2rss.xlab.app/feed/567cb1a8cf49f3e2c141d9d8085712f42ffc2fef.xml",
    "信息安全国家工程研究中心":"https://wechat2rss.xlab.app/feed/7caad9bdb6b168fe174bc815a9b44b7f52d7198b.xml",
    "关键基础设施安全应急响应中心":"https://wechat2rss.xlab.app/feed/567cb1a8cf49f3e2c141d9d8085712f42ffc2fef.xml"
}

class RawArticle(BaseModel):
    """当前爬虫原始数据结构"""
    web_url: str = Field(..., description="原始文章链接")
    title: str = Field(..., description="原始文章标题")
    trail_text: Optional[str] = Field(None, description="原始导语或摘要")
    body_text: Optional[str] = Field(None, description="原始正文全文")
    web_publication_date: Optional[str] = Field(None, description="原始发布时间")
    section_name: Optional[str] = Field(None, description="所属新闻版块")
    api_url: Optional[str] = Field(None, description="API 请求链接")


def clean_html(raw_html: str) -> str:
    """去除 HTML 标签"""
    if not raw_html:
        return None
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def fetch_article_body(url: str) -> Optional[str]:
    """
    尝试抓取正文（可选）
    对微信源有时有效，有时会被反爬
    """
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
        resp.raise_for_status()
        return clean_html(resp.text)
    except Exception:
        return None


def parse_rss_feed(feed_name: str, rss_url: str, fetch_body: bool = False) -> List[RawArticle]:
    """
    解析单个 RSS 源
    """
    parsed = feedparser.parse(rss_url)
    articles = []

    for entry in parsed.entries:
        web_url = entry.get("link")
        title = entry.get("title")
        summary = clean_html(entry.get("summary", ""))
        pub_date = entry.get("published") or entry.get("pubDate")

        body_text = None

        # RSS 有 content 字段时优先
        if "content" in entry and entry.content:
            body_text = clean_html(entry.content[0].value)

        # 没有则尝试爬原网页
        elif fetch_body and web_url:
            body_text = fetch_article_body(web_url)

        article = RawArticle(
            web_url=web_url,
            title=title,
            trail_text=summary,
            body_text=body_text,
            web_publication_date=pub_date,
            section_name=feed_name,
            api_url=rss_url
        )

        articles.append(article)

    return articles


def parse_pool(pool: dict, fetch_body: bool = False) -> List[RawArticle]:
    """
    解析整个 RSS 池
    """
    all_articles = []

    for name, url in pool.items():
        try:
            articles = parse_rss_feed(name, url, fetch_body)
            all_articles.extend(articles)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    return all_articles



if __name__ == '__main__':
    articles = parse_pool(pool, fetch_body=False)
    for a in articles[:3]:
        print(a.model_dump())