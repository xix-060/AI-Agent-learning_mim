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
    tags        TEXT    NOT NULL DEFAULT '',
    completed   INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_tags(tags: str | list[str] | None) -> str:
    """将标签输入规范化为逗号分隔字符串。

    Args:
        tags: 原始标签，支持逗号分隔字符串或字符串列表。

    Returns:
        去重、去空白后的逗号分隔标签字符串。
    """
    if tags is None:
        return ""

    if isinstance(tags, list):
        raw_items = tags
    else:
        raw_items = str(tags).split(",")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = item.strip()
        if tag and tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return ",".join(normalized)


def parse_tags(tags: str) -> list[str]:
    """将逗号分隔标签字符串解析为列表。

    Args:
        tags: 数据库存储的标签字符串。

    Returns:
        标签列表；空字符串返回空列表。
    """
    if not tags.strip():
        return []
    return [part.strip() for part in tags.split(",") if part.strip()]


def task_has_tag(tags: str, tag: str) -> bool:
    """判断任务是否包含指定标签。

    Args:
        tags: 任务标签字符串。
        tag: 待匹配的标签。

    Returns:
        包含指定标签返回 True。
    """
    target = tag.strip()
    if not target:
        return False
    return target in parse_tags(tags)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将数据库行转换为 API 友好的字典。"""
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "tags": row["tags"],
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

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """迁移旧版数据库 schema，补齐 tags 字段。"""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        if "tags" not in columns:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
            )

    def init_db(self) -> None:
        """初始化数据库表结构。"""
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
            self._migrate_schema(conn)

    def create_task(
        self,
        title: str,
        description: str = "",
        tags: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """创建新任务。

        Args:
            title: 任务标题。
            description: 任务描述。
            tags: 任务标签，逗号分隔字符串或列表。

        Returns:
            新创建的任务字典。
        """
        now = _now_iso()
        normalized_tags = normalize_tags(tags)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks (title, description, tags, completed, created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, ?)
                """,
                (title, description, normalized_tags, now, now),
            )
            task_id = cursor.lastrowid
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
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
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def list_tasks(
        self,
        status: TaskStatus = "all",
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """获取任务列表，可按状态与标签筛选。

        Args:
            status: 筛选状态，``all``、``pending`` 或 ``completed``。
            tag: 单个标签筛选；为空时不按标签过滤。

        Returns:
            任务字典列表，按 ID 升序排列。
        """
        query = "SELECT * FROM tasks"
        conditions: list[str] = []
        params: list[Any] = []

        if status == "pending":
            conditions.append("completed = 0")
        elif status == "completed":
            conditions.append("completed = 1")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        tasks = [_row_to_dict(row) for row in rows]
        if tag is not None and tag.strip():
            tasks = [task for task in tasks if task_has_tag(task["tags"], tag)]
        return tasks

    def update_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        tags: str | list[str] | None = None,
        completed: bool | None = None,
    ) -> dict[str, Any] | None:
        """部分更新任务字段。

        Args:
            task_id: 任务 ID。
            title: 新标题，None 表示不更新。
            description: 新描述，None 表示不更新。
            tags: 新标签，None 表示不更新。
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
        new_tags = normalize_tags(tags) if tags is not None else current["tags"]
        new_completed = (
            int(completed) if completed is not None else int(current["completed"])
        )
        now = _now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET title = ?, description = ?, tags = ?, completed = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, new_description, new_tags, new_completed, now, task_id),
            )
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
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

    def get_stats(self) -> dict[str, Any]:
        """获取任务统计数据。

        Returns:
            包含 total、pending、completed、tag_counts 的统计字典。
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE completed = 1",
            ).fetchone()[0]
            rows = conn.execute("SELECT tags FROM tasks").fetchall()

        pending = total - completed
        tag_counts: dict[str, int] = {}
        for row in rows:
            for tag in parse_tags(row["tags"]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total": total,
            "pending": pending,
            "completed": completed,
            "tag_counts": tag_counts,
        }
