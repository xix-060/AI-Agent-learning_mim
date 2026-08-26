"""Todo 应用 Flask 入口与 API 路由。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from todo_app.db import DEFAULT_DB_PATH, TodoDatabase

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def create_app(db_path: str | Path | None = None) -> Flask:
    """创建并配置 Flask 应用。

    Args:
        db_path: 可选的数据库路径，默认使用 ``todo_app/data/todo.db``。

    Returns:
        配置完成的 Flask 应用实例。
    """
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    database = TodoDatabase(db_path or DEFAULT_DB_PATH)
    app.config["todo_db"] = database

    @app.get("/")
    def index() -> str:
        """渲染 Todo 单页前端。"""
        return render_template("index.html")

    @app.get("/api/tasks")
    def list_tasks() -> tuple[Any, int]:
        """获取任务列表，支持按状态筛选。"""
        status = request.args.get("status", "all")
        if status not in ("all", "pending", "completed"):
            return jsonify(
                {"error": "status 参数无效，可选 all/pending/completed"}
            ), 400

        tasks = database.list_tasks(status=status)  # type: ignore[arg-type]
        return jsonify(tasks), 200

    @app.post("/api/tasks")
    def create_task() -> tuple[Any, int]:
        """创建新任务。"""
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title", "")).strip()
        description = str(payload.get("description", "")).strip()

        if not title:
            return jsonify({"error": "title 不能为空"}), 400

        task = database.create_task(title=title, description=description)
        return jsonify(task), 201

    @app.get("/api/tasks/<int:task_id>")
    def get_task(task_id: int) -> tuple[Any, int]:
        """获取单条任务详情。"""
        task = database.get_task(task_id)
        if task is None:
            return jsonify({"error": "任务不存在"}), 404
        return jsonify(task), 200

    @app.patch("/api/tasks/<int:task_id>")
    def update_task(task_id: int) -> tuple[Any, int]:
        """部分更新任务。"""
        payload = request.get_json(silent=True) or {}

        title: str | None = None
        if "title" in payload:
            title = str(payload["title"]).strip()
            if not title:
                return jsonify({"error": "title 不能为空"}), 400

        description: str | None = None
        if "description" in payload:
            description = str(payload["description"]).strip()

        completed: bool | None = None
        if "completed" in payload:
            completed = bool(payload["completed"])

        if title is None and description is None and completed is None:
            return jsonify({"error": "至少提供一个可更新字段"}), 400

        task = database.update_task(
            task_id,
            title=title,
            description=description,
            completed=completed,
        )
        if task is None:
            return jsonify({"error": "任务不存在"}), 404
        return jsonify(task), 200

    @app.delete("/api/tasks/<int:task_id>")
    def delete_task(task_id: int) -> tuple[Any, int]:
        """删除任务。"""
        deleted = database.delete_task(task_id)
        if not deleted:
            return jsonify({"error": "任务不存在"}), 404
        return "", 204

    @app.get("/api/stats")
    def get_stats() -> tuple[Any, int]:
        """获取任务统计数据。"""
        return jsonify(database.get_stats()), 200

    return app


def main() -> None:
    """以开发模式启动 Flask 服务。"""
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
