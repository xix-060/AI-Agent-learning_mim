"""Todo 应用 API 单元测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from flask.testing import FlaskClient

from todo_app.app import create_app


@pytest.fixture
def client(tmp_path) -> FlaskClient:
    """创建使用临时数据库的 Flask 测试客户端。"""
    db_path = tmp_path / "test_todo.db"
    app = create_app(db_path=db_path)
    app.config["TESTING"] = True
    return app.test_client()


def _post_task(
    client: FlaskClient, title: str, description: str = ""
) -> dict[str, Any]:
    """辅助函数：创建任务并返回响应 JSON。"""
    response = client.post(
        "/api/tasks",
        data=json.dumps({"title": title, "description": description}),
        content_type="application/json",
    )
    return response.get_json()


def test_create_task(client: FlaskClient) -> None:
    """创建任务应返回 201 及完整任务对象。"""
    response = client.post(
        "/api/tasks",
        data=json.dumps({"title": "学习 Flask", "description": "完成 Todo 后端"}),
        content_type="application/json",
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["title"] == "学习 Flask"
    assert data["description"] == "完成 Todo 后端"
    assert data["completed"] is False
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_create_task_empty_title_returns_400(client: FlaskClient) -> None:
    """空标题应返回 400 及错误信息。"""
    response = client.post(
        "/api/tasks",
        data=json.dumps({"title": "   ", "description": "无效任务"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_list_tasks(client: FlaskClient) -> None:
    """应返回所有任务列表。"""
    _post_task(client, "任务 A")
    _post_task(client, "任务 B")

    response = client.get("/api/tasks")

    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert data[0]["title"] == "任务 A"
    assert data[1]["title"] == "任务 B"


def test_list_tasks_filter_pending(client: FlaskClient) -> None:
    """按进行中状态筛选任务。"""
    _post_task(client, "进行中任务")
    task_b = _post_task(client, "已完成任务")
    client.patch(
        f"/api/tasks/{task_b['id']}",
        data=json.dumps({"completed": True}),
        content_type="application/json",
    )

    response = client.get("/api/tasks?status=pending")

    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "进行中任务"
    assert data[0]["completed"] is False


def test_list_tasks_filter_completed(client: FlaskClient) -> None:
    """按已完成状态筛选任务。"""
    task_a = _post_task(client, "待完成任务")
    _post_task(client, "另一个任务")
    client.patch(
        f"/api/tasks/{task_a['id']}",
        data=json.dumps({"completed": True}),
        content_type="application/json",
    )

    response = client.get("/api/tasks?status=completed")

    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]["title"] == "待完成任务"
    assert data[0]["completed"] is True


def test_get_task(client: FlaskClient) -> None:
    """获取单条任务详情。"""
    created = _post_task(client, "单条任务", "描述信息")

    response = client.get(f"/api/tasks/{created['id']}")

    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == created["id"]
    assert data["title"] == "单条任务"
    assert data["description"] == "描述信息"


def test_get_task_not_found(client: FlaskClient) -> None:
    """不存在的任务应返回 404。"""
    response = client.get("/api/tasks/9999")

    assert response.status_code == 404
    assert "error" in response.get_json()


def test_update_task(client: FlaskClient) -> None:
    """部分更新任务标题、描述与完成状态。"""
    created = _post_task(client, "旧标题", "旧描述")

    response = client.patch(
        f"/api/tasks/{created['id']}",
        data=json.dumps(
            {"title": "新标题", "description": "新描述", "completed": True},
        ),
        content_type="application/json",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["title"] == "新标题"
    assert data["description"] == "新描述"
    assert data["completed"] is True


def test_update_task_empty_title_returns_400(client: FlaskClient) -> None:
    """更新为空标题应返回 400。"""
    created = _post_task(client, "有效标题")

    response = client.patch(
        f"/api/tasks/{created['id']}",
        data=json.dumps({"title": ""}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_update_task_not_found(client: FlaskClient) -> None:
    """更新不存在的任务应返回 404。"""
    response = client.patch(
        "/api/tasks/9999",
        data=json.dumps({"title": "不存在"}),
        content_type="application/json",
    )

    assert response.status_code == 404


def test_delete_task(client: FlaskClient) -> None:
    """删除任务应返回 204，且列表中不再存在。"""
    created = _post_task(client, "待删除任务")

    response = client.delete(f"/api/tasks/{created['id']}")

    assert response.status_code == 204
    list_response = client.get("/api/tasks")
    assert list_response.get_json() == []


def test_delete_task_not_found(client: FlaskClient) -> None:
    """删除不存在的任务应返回 404。"""
    response = client.delete("/api/tasks/9999")

    assert response.status_code == 404


def test_stats(client: FlaskClient) -> None:
    """统计接口应返回总数、进行中与已完成数量。"""
    task_a = _post_task(client, "任务 1")
    _post_task(client, "任务 2")
    _post_task(client, "任务 3")
    client.patch(
        f"/api/tasks/{task_a['id']}",
        data=json.dumps({"completed": True}),
        content_type="application/json",
    )

    response = client.get("/api/stats")

    assert response.status_code == 200
    data = response.get_json()
    assert data == {"total": 3, "pending": 2, "completed": 1}
