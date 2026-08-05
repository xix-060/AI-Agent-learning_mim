# LangGraph 进阶

## 1. Checkpointer（持久化）

- MemorySaver：内存存储（重启丢失）
- SqliteSaver：SQLite 存储（本地持久化）
- PostgresSaver：生产级

## 2. 可暂停/可恢复

- interrupt()：主动暂停
- thread\_id：会话隔离
- invoke(None, config)：从暂停点恢复

## 3. Human-in-the-Loop 三种模式

1. 审批模式：危险操作前问人（今天实现）
2. 编辑模式：人修改 Agent 的输出
3. 引导模式：人提供额外信息

## 4. 应用场景

- 文件操作（删除/覆盖前确认）
- 发邮件/消息（发送前确认）
- 付费操作（下单前确认）
- 数据库修改（写入前确认）
