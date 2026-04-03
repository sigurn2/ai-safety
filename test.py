import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.technologyreview.com/2026/03/30/1134881/the-pentagons-culture-war-tactic-against-anthropic-has-backfired/",
        )
        print(result.markdown)  # type: ignore

if __name__ == "__main__":
    asyncio.run(main())
