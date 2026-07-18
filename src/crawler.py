"""Hacker News 爬虫 - 默认通过 Algolia HN Search API 获取首页数据（国内可直连）。"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

DEFAULT_URL = "https://news.ycombinator.com/"
DEFAULT_ALGOLIA_API = "https://hn.algolia.com/api/v1/search"
DEFAULT_OUTPUT = Path("data/hacker_news.json")
DEFAULT_DOMAIN_REPORT = Path("docs/news_domains.md")
DEFAULT_HITS_PER_PAGE = 30
TOP_STORIES_FOR_DOMAIN = 10
REQUEST_TIMEOUT = 15
USER_AGENT = (
    "AI-Agent-learning_mim/1.0 (+https://github.com/xix-060/AI-Agent-learning_mim)"
)


class DataSource(str, Enum):
    """数据来源。"""

    ALGOLIA = "algolia"
    HTML = "html"


class HackerNewsItem(BaseModel):
    """Hacker News 首页单条新闻。"""

    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


def fetch_page(url: str = DEFAULT_URL, timeout: int = REQUEST_TIMEOUT) -> str:
    """请求 HN 官网页面并返回 HTML 文本。

    Args:
        url: 目标页面 URL。
        timeout: 请求超时时间（秒）。

    Returns:
        页面 HTML 字符串。

    Raises:
        requests.HTTPError: 响应状态码非 2xx。
        requests.RequestException: 网络或连接异常。
    """
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def fetch_algolia_stories(
    api_url: str = DEFAULT_ALGOLIA_API,
    hits_per_page: int = DEFAULT_HITS_PER_PAGE,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """从 Algolia HN Search API 获取首页故事 JSON。

    Algolia 是 Hacker News 官方搜索服务，国内网络通常可直连，
    比直接访问 news.ycombinator.com 更稳定。

    Args:
        api_url: Algolia 搜索 API 地址。
        hits_per_page: 返回条目数量。
        timeout: 请求超时时间（秒）。

    Returns:
        Algolia API 原始 JSON 响应。

    Raises:
        requests.HTTPError: 响应状态码非 2xx。
        requests.RequestException: 网络或连接异常。
    """
    response = requests.get(
        api_url,
        params={"tags": "front_page", "hitsPerPage": hits_per_page},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def parse_stories(html: str, base_url: str = DEFAULT_URL) -> list[HackerNewsItem]:
    """从 HN 官网 HTML 中解析首页新闻标题与链接。

    Args:
        html: 页面 HTML 内容。
        base_url: 用于补全相对链接的基础 URL。

    Returns:
        解析得到的新闻列表，顺序与页面一致。
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[HackerNewsItem] = []

    for row in soup.select("tr.athing"):
        link = row.select_one("span.titleline > a")
        if link is None:
            continue

        title = link.get_text(strip=True)
        href = link.get("href", "").strip()
        if not title or not href:
            continue

        items.append(HackerNewsItem(title=title, url=urljoin(base_url, href)))

    return items


def parse_algolia_stories(payload: dict[str, Any]) -> list[HackerNewsItem]:
    """解析 Algolia API 响应为新闻列表。

    Args:
        payload: Algolia API 返回的 JSON 对象。

    Returns:
        解析得到的新闻列表。
    """
    items: list[HackerNewsItem] = []

    for hit in payload.get("hits", []):
        title = str(hit.get("title", "")).strip()
        if not title:
            continue

        story_url = hit.get("url")
        if story_url:
            url = str(story_url).strip()
        else:
            story_id = hit.get("objectID") or hit.get("story_id")
            if not story_id:
                continue
            url = f"{DEFAULT_URL}item?id={story_id}"

        items.append(HackerNewsItem(title=title, url=url))

    return items


def save_stories(
    items: list[HackerNewsItem], path: str | Path = DEFAULT_OUTPUT
) -> Path:
    """将新闻列表保存为 JSON 文件。

    Args:
        items: 待保存的新闻列表。
        path: 输出文件路径。

    Returns:
        写入后的文件路径。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = [item.model_dump() for item in items]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def extract_domain(url: str) -> str:
    """从 URL 中提取域名，并去除 ``www.`` 前缀。

    Args:
        url: 新闻链接。

    Returns:
        规范化后的域名；无法解析时返回 ``unknown``。
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return "unknown"

    hostname = hostname.lower()
    if hostname.startswith("www."):
        return hostname[4:]
    return hostname


def count_domain_distribution(
    items: list[HackerNewsItem],
    top_n: int = TOP_STORIES_FOR_DOMAIN,
) -> dict[str, int]:
    """统计前 N 条新闻的域名出现次数。

    Args:
        items: 新闻列表。
        top_n: 参与统计的条目数量。

    Returns:
        按出现次数降序排列的域名字典。
    """
    domains = [extract_domain(item.url) for item in items[:top_n]]
    counter = Counter(domains)
    return dict(counter.most_common())


def format_domain_report(
    items: list[HackerNewsItem],
    distribution: dict[str, int],
    top_n: int = TOP_STORIES_FOR_DOMAIN,
) -> str:
    """将域名分布格式化为 Markdown 文本。

    Args:
        items: 新闻列表。
        distribution: 域名计数字典。
        top_n: 参与统计的条目数量。

    Returns:
        Markdown 格式报告内容。
    """
    selected_items = items[:top_n]
    total = len(selected_items)
    lines = [
        "# Hacker News 前 10 条新闻域名分布",
        "",
        f"统计范围：前 {top_n} 条新闻",
        f"新闻总数：{total}",
        "",
        "| 域名 | 数量 | 占比 |",
        "| --- | ---: | ---: |",
    ]

    for domain, count in distribution.items():
        percentage = f"{count / total * 100:.1f}%" if total else "0.0%"
        lines.append(f"| {domain} | {count} | {percentage} |")

    lines.extend(["", "## 新闻列表", ""])
    for index, item in enumerate(selected_items, start=1):
        domain = extract_domain(item.url)
        lines.append(f"{index}. [{item.title}]({item.url}) — `{domain}`")

    lines.append("")
    return "\n".join(lines)


def save_domain_report(
    items: list[HackerNewsItem],
    path: str | Path = DEFAULT_DOMAIN_REPORT,
    top_n: int = TOP_STORIES_FOR_DOMAIN,
) -> Path:
    """统计前 N 条新闻域名分布并写入 Markdown 文件。

    Args:
        items: 新闻列表。
        path: Markdown 输出路径。
        top_n: 参与统计的条目数量。

    Returns:
        写入后的文件路径。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distribution = count_domain_distribution(items, top_n=top_n)
    content = format_domain_report(items, distribution, top_n=top_n)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def crawl_hacker_news(
    output_path: str | Path = DEFAULT_OUTPUT,
    domain_report_path: str | Path = DEFAULT_DOMAIN_REPORT,
    source: DataSource = DataSource.ALGOLIA,
    url: str = DEFAULT_URL,
    api_url: str = DEFAULT_ALGOLIA_API,
    hits_per_page: int = DEFAULT_HITS_PER_PAGE,
    timeout: int = REQUEST_TIMEOUT,
) -> list[HackerNewsItem]:
    """爬取 Hacker News 首页并保存到 JSON。

    默认使用 Algolia HN Search API（国内可直连）；若网络可访问官网，
    可设置 ``source=DataSource.HTML`` 改走 HTML 解析。

    Args:
        output_path: JSON 输出路径。
        domain_report_path: 域名分布 Markdown 输出路径。
        source: 数据来源，``algolia`` 或 ``html``。
        url: HN 官网 URL，仅在 ``source=html`` 时使用。
        api_url: Algolia API 地址，仅在 ``source=algolia`` 时使用。
        hits_per_page: Algolia 返回条目数量。
        timeout: 请求超时时间（秒）。

    Returns:
        爬取到的新闻列表。
    """
    if source is DataSource.ALGOLIA:
        payload = fetch_algolia_stories(
            api_url=api_url,
            hits_per_page=hits_per_page,
            timeout=timeout,
        )
        items = parse_algolia_stories(payload)
    else:
        html = fetch_page(url, timeout=timeout)
        items = parse_stories(html, base_url=url)

    save_stories(items, output_path)
    save_domain_report(items, domain_report_path)
    return items


def main() -> None:
    """命令行入口：爬取首页、保存 JSON，并生成域名分布报告。"""
    items = crawl_hacker_news()
    print(f"已通过 Algolia API 保存 {len(items)} 条新闻到 {DEFAULT_OUTPUT}")
    print(f"已生成域名分布报告到 {DEFAULT_DOMAIN_REPORT}")


if __name__ == "__main__":
    main()
