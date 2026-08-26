# 个人 Todo 应用

轻量级个人任务管理应用，技术栈为 **Flask + SQLite + 原生 HTML/JS 单页**。

## 功能

- 任务的增删改查（CRUD）
- 标记完成 / 取消完成
- 按状态筛选（全部 / 进行中 / 已完成）
- 顶部统计条（总数 / 进行中 / 已完成）
- 任务描述字段（可选）

## 目录结构

```
todo_app/
├── app.py              # Flask 应用入口与 API 路由
├── db.py               # SQLite 数据库封装
├── templates/
│   └── index.html      # 单页前端（HTML + CSS + 原生 JS）
├── data/
│   └── todo.db         # SQLite 数据库（首次运行自动创建）
└── README.md
```

测试文件位于仓库根目录：`tests/test_api.py`

## 快速开始

### 1. 安装依赖

```bash
pip install flask
```

或从项目根目录安装全部依赖：

```bash
pip install -r requirements.txt
```

### 2. 启动服务

在项目根目录执行：

```bash
python -m todo_app.app
```

浏览器访问：<http://127.0.0.1:5000>

### 3. 运行测试

```bash
pytest tests/test_api.py -v
```

## 数据库

单表 `tasks`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 主键，自增 |
| `title` | TEXT | 任务标题（必填） |
| `description` | TEXT | 任务描述（默认空字符串） |
| `completed` | INTEGER | 0=进行中，1=已完成 |
| `created_at` | TEXT | 创建时间（ISO8601 UTC） |
| `updated_at` | TEXT | 更新时间（ISO8601 UTC） |

数据库文件默认路径：`todo_app/data/todo.db`

## API 文档

统一前缀 `/api`，请求与响应均为 JSON。

### 获取任务列表

```
GET /api/tasks?status=all|pending|completed
```

`status` 默认为 `all`。

### 创建任务

```
POST /api/tasks
Content-Type: application/json

{
  "title": "学习 Flask",
  "description": "完成 Todo 后端"
}
```

- 成功：`201`
- 标题为空：`400`

### 获取单条任务

```
GET /api/tasks/<id>
```

- 不存在：`404`

### 部分更新任务

```
PATCH /api/tasks/<id>
Content-Type: application/json

{
  "title": "新标题",
  "description": "新描述",
  "completed": true
}
```

字段均为可选，至少提供一个。标题不能为空。

### 删除任务

```
DELETE /api/tasks/<id>
```

- 成功：`204`（无响应体）
- 不存在：`404`

### 获取统计

```
GET /api/stats
```

响应示例：

```json
{
  "total": 10,
  "pending": 6,
  "completed": 4
}
```

## 开发说明

- 所有 SQL 使用参数化查询，防止 SQL 注入
- 前端使用 `textContent` 渲染用户数据，防止 XSS
- 测试使用 Flask test client + 临时 SQLite 数据库，互不干扰
- 当前为开发模式（`debug=True`），仅供本地学习使用

## 技术选型

| 层级 | 技术 |
|------|------|
| 后端 | Flask |
| 数据库 | SQLite（标准库） |
| 前端 | 原生 HTML / CSS / JavaScript |
| 测试 | pytest |
