"""Todo 应用 SQLite 数据库封装。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Literal

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "todo.db"

TaskStatus = Literal["all", "pending", "completed"]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    completed   INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将数据库行转换为 API 友好的字典。"""
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "completed": bool(row["completed"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


class TodoDatabase:
    """Todo SQLite 数据库操作类。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """初始化数据库路径并确保表结构存在。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接上下文管理器。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        """初始化数据库表结构。"""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def create_task(self, title: str, description: str = "") -> dict[str, Any]:
        """创建新任务。

        Args:
            title: 任务标题。
            description: 任务描述。

        Returns:
            新创建的任务字典。
        """
        now = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks (title, description, completed, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?)
                """,
                (title, description, now, now),
            )
            task_id = cursor.lastrowid
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _row_to_dict(row)

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        """根据 ID 获取单条任务。

        Args:
            task_id: 任务 ID。

        Returns:
            任务字典；不存在时返回 None。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def list_tasks(self, status: TaskStatus = "all") -> list[dict[str, Any]]:
        """获取任务列表，可按状态筛选。

        Args:
            status: 筛选状态，``all``、``pending`` 或 ``completed``。

        Returns:
            任务字典列表，按 ID 升序排列。
        """
        query = "SELECT * FROM tasks"
        params: tuple[Any, ...] = ()

        if status == "pending":
            query += " WHERE completed = 0"
        elif status == "completed":
            query += " WHERE completed = 1"

        query += " ORDER BY id ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        completed: bool | None = None,
    ) -> dict[str, Any] | None:
        """部分更新任务字段。

        Args:
            task_id: 任务 ID。
            title: 新标题，None 表示不更新。
            description: 新描述，None 表示不更新。
            completed: 新完成状态，None 表示不更新。

        Returns:
            更新后的任务字典；不存在时返回 None。
        """
        current = self.get_task(task_id)
        if current is None:
            return None

        new_title = title if title is not None else current["title"]
        new_description = (
            description if description is not None else current["description"]
        )
        new_completed = (
            int(completed) if completed is not None else int(current["completed"])
        )
        now = _now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET title = ?, description = ?, completed = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_description, new_completed, now, task_id),
            )
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _row_to_dict(row)

    def delete_task(self, task_id: int) -> bool:
        """删除任务。

        Args:
            task_id: 任务 ID。

        Returns:
            删除成功返回 True，任务不存在返回 False。
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0

    def get_stats(self) -> dict[str, int]:
        """获取任务统计数据。

        Returns:
            包含 total、pending、completed 的统计字典。
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE completed = 1",
            ).fetchone()[0]
        pending = total - completed
        return {"total": total, "pending": pending, "completed": completed}
