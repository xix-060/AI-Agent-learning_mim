"""crawler 模块单元测试。"""

import json
from unittest.mock import patch

import pytest
import requests

from src.crawler import (
    DEFAULT_ALGOLIA_API,
    DEFAULT_URL,
    DataSource,
    HackerNewsItem,
    crawl_hacker_news,
    fetch_algolia_stories,
    fetch_page,
    parse_algolia_stories,
    parse_stories,
    save_stories,
)

SAMPLE_HTML = """
<html>
  <body>
    <table>
      <tr class="athing" id="1">
        <td class="title">
          <span class="titleline">
            <a href="https://example.com/story-one">Story One</a>
          </span>
        </td>
      </tr>
      <tr class="athing" id="2">
        <td class="title">
          <span class="titleline">
            <a href="item?id=12345">Ask HN: Example Question</a>
          </span>
        </td>
      </tr>
      <tr class="athing" id="3">
        <td class="title">
          <span class="titleline"></span>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

SAMPLE_ALGOLIA_PAYLOAD = {
    "hits": [
        {
            "title": "Story One",
            "url": "https://example.com/story-one",
            "objectID": "100",
        },
        {
            "title": "Ask HN: Example Question",
            "url": None,
            "objectID": "12345",
        },
        {
            "title": "",
            "url": "https://example.com/ignored",
            "objectID": "999",
        },
    ]
}


def test_parse_stories_extracts_title_and_url():
    items = parse_stories(SAMPLE_HTML)

    assert len(items) == 2
    assert items[0] == HackerNewsItem(
        title="Story One", url="https://example.com/story-one"
    )
    assert items[1] == HackerNewsItem(
        title="Ask HN: Example Question",
        url="https://news.ycombinator.com/item?id=12345",
    )


def test_parse_stories_empty_html():
    assert parse_stories("<html><body></body></html>") == []


def test_parse_algolia_stories_extracts_title_and_url():
    items = parse_algolia_stories(SAMPLE_ALGOLIA_PAYLOAD)

    assert len(items) == 2
    assert items[0] == HackerNewsItem(
        title="Story One", url="https://example.com/story-one"
    )
    assert items[1] == HackerNewsItem(
        title="Ask HN: Example Question",
        url="https://news.ycombinator.com/item?id=12345",
    )


def test_parse_algolia_stories_empty_payload():
    assert parse_algolia_stories({"hits": []}) == []


def test_save_stories_writes_json(tmp_path):
    items = [
        HackerNewsItem(title="Story One", url="https://example.com/story-one"),
        HackerNewsItem(title="Story Two", url="https://example.com/story-two"),
    ]
    output_path = tmp_path / "hacker_news.json"

    saved_path = save_stories(items, output_path)

    assert saved_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == [
        {"title": "Story One", "url": "https://example.com/story-one"},
        {"title": "Story Two", "url": "https://example.com/story-two"},
    ]


@patch("src.crawler.requests.get")
def test_fetch_page_returns_html(mock_get):
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.text = "<html>ok</html>"
    mock_response.apparent_encoding = "utf-8"
    mock_response.raise_for_status.return_value = None

    html = fetch_page(DEFAULT_URL)

    assert html == "<html>ok</html>"
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["timeout"] == 15


@patch("src.crawler.requests.get")
def test_fetch_page_raises_on_http_error(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")

    with pytest.raises(requests.HTTPError):
        fetch_page(DEFAULT_URL)


@patch("src.crawler.requests.get")
def test_fetch_algolia_stories_returns_json(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = SAMPLE_ALGOLIA_PAYLOAD

    payload = fetch_algolia_stories()

    assert payload == SAMPLE_ALGOLIA_PAYLOAD
    mock_get.assert_called_once_with(
        DEFAULT_ALGOLIA_API,
        params={"tags": "front_page", "hitsPerPage": 30},
        timeout=15,
        headers={"User-Agent": mock_get.call_args.kwargs["headers"]["User-Agent"]},
    )


@patch("src.crawler.save_stories")
@patch("src.crawler.fetch_algolia_stories")
def test_crawl_hacker_news_algolia(mock_fetch_algolia, mock_save_stories):
    mock_fetch_algolia.return_value = SAMPLE_ALGOLIA_PAYLOAD
    mock_save_stories.return_value = "data/hacker_news.json"

    items = crawl_hacker_news(output_path="data/hacker_news.json")

    assert len(items) == 2
    mock_fetch_algolia.assert_called_once()
    mock_save_stories.assert_called_once()


@patch("src.crawler.save_stories")
@patch("src.crawler.fetch_page")
def test_crawl_hacker_news_html(mock_fetch_page, mock_save_stories):
    mock_fetch_page.return_value = SAMPLE_HTML
    mock_save_stories.return_value = "data/hacker_news.json"

    items = crawl_hacker_news(
        output_path="data/hacker_news.json", source=DataSource.HTML
    )

    assert len(items) == 2
    mock_fetch_page.assert_called_once_with(DEFAULT_URL, timeout=15)
    mock_save_stories.assert_called_once()
