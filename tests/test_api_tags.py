"""Todo 应用标签功能 API 单元测试（SPEC_TAGS M1）。"""

from __future__ import annotations

import json

import pytest
from flask.testing import FlaskClient

from todo_app.app import create_app
from todo_app.db import TodoDatabase


@pytest.fixture
def client(tmp_path) -> FlaskClient:
    """创建使用临时数据库的 Flask 测试客户端。"""
    db_path = tmp_path / "test_todo_tags.db"
    app = create_app(db_path=db_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_create_task_with_tags(client: FlaskClient, tmp_path) -> None:
    """带标签创建任务，API 与 DB 中 tags 字段正确。"""
    response = client.post(
        "/api/tasks",
        data=json.dumps({"title": "写报告", "tags": "工作,学习"}),
        content_type="application/json",
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["tags"] == "工作,学习"

    db = TodoDatabase(tmp_path / "test_todo_tags.db")
    stored = db.get_task(data["id"])
    assert stored is not None
    assert stored["tags"] == "工作,学习"


def test_create_task_without_tags_defaults_empty(client: FlaskClient) -> None:
    """不传 tags 字段时不报错，默认空字符串。"""
    response = client.post(
        "/api/tasks",
        data=json.dumps({"title": "无标签任务"}),
        content_type="application/json",
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["tags"] == ""


def test_list_tasks_filter_by_tag(client: FlaskClient) -> None:
    """按单个标签筛选只返回匹配任务。"""
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "工作任务", "tags": "工作"}),
        content_type="application/json",
    )
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "学习任务", "tags": "学习"}),
        content_type="application/json",
    )
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "混合任务", "tags": "工作,生活"}),
        content_type="application/json",
    )

    response = client.get("/api/tasks?tag=工作")

    assert response.status_code == 200
    titles = [item["title"] for item in response.get_json()]
    assert titles == ["工作任务", "混合任务"]


def test_list_tasks_without_tag_includes_empty_tags(client: FlaskClient) -> None:
    """无标签筛选时，标签为空的任务仍在全部列表中显示。"""
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "无标签任务"}),
        content_type="application/json",
    )
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "有标签任务", "tags": "生活"}),
        content_type="application/json",
    )

    response = client.get("/api/tasks")

    assert response.status_code == 200
    titles = [item["title"] for item in response.get_json()]
    assert "无标签任务" in titles
    assert "有标签任务" in titles


def test_update_task_tags(client: FlaskClient) -> None:
    """PATCH 可更新任务标签。"""
    created = client.post(
        "/api/tasks",
        data=json.dumps({"title": "旧任务", "tags": "工作"}),
        content_type="application/json",
    ).get_json()

    response = client.patch(
        f"/api/tasks/{created['id']}",
        data=json.dumps({"tags": "学习,生活"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.get_json()["tags"] == "学习,生活"


def test_stats_includes_tag_counts(client: FlaskClient) -> None:
    """统计接口包含各标签数量。"""
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "任务 1", "tags": "工作,学习"}),
        content_type="application/json",
    )
    client.post(
        "/api/tasks",
        data=json.dumps({"title": "任务 2", "tags": "工作"}),
        content_type="application/json",
    )

    response = client.get("/api/stats")

    assert response.status_code == 200
    data = response.get_json()
    assert data["tag_counts"] == {"工作": 2, "学习": 1}
