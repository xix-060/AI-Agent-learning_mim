# Linux 常用命令速查表

---

## 1. 文件操作

```bash
# ls — 列出目录
ls -la              # 详细 + 隐藏（最常用）
ls -lh              # 人类可读大小
ls -lt              # 按时间排序（新→旧）
ls -R               # 递归子目录

# cd — 切换目录
cd /home/user       # 绝对路径
cd ..               # 上级目录
cd ~                # 主目录
cd -                # 回到上次目录

# cp — 复制
cp file.txt /tmp/               # 复制文件
cp -r mydir /backup/            # 递归复制目录
cp -i file.txt /tmp/            # 覆盖前确认
cp -p file.txt /tmp/            # 保留权限和时间戳

# mv — 移动/重命名
mv old.txt new.txt              # 重命名
mv file.txt /tmp/               # 移动
mv -i file.txt /tmp/            # 覆盖前确认

# rm — 删除
rm file.txt                     # 删除文件
rm -rf mydir                    # 递归强制删除目录（⚠️ 慎用！）
rm -i *.tmp                     # 删除前确认（推荐）

# find — 查找文件
find . -name "*.py"             # 按名称查找
find /var/log -name "*.log" -mtime -7   # 7天内修改的日志
find . -type f -size +100M      # 大于100MB的文件
find . -name "*.pyc" -delete    # 找到并删除
find . -maxdepth 2 -name "*.yml" # 限制搜索深度

# tree — 树形显示
tree -L 2                       # 只显示2层
tree -d                         # 只显示目录
tree -I "node_modules|.git"     # 排除指定目录
```

---

## 2. 文本处理

```bash
# cat — 查看文件
cat -n file.txt                 # 带行号
cat -s file.txt                  # 压缩连续空行
cat file1 file2 > merged.txt     # 合并文件

# grep — 文本搜索
grep "error" app.log            # 搜索关键词
grep -i "error" app.log         # 忽略大小写
grep -rn "TODO" ./src/          # 递归 + 行号
grep -v "debug" app.log         # 反向匹配（排除）
grep -c "error" app.log         # 只输出匹配行数
grep -E "error|warning" log     # 正则多关键词
grep -A 3 "error" log           # 匹配行 + 后3行

# awk — 列处理
awk '{print $1}' file.txt       # 打印第1列
awk -F',' '{print $2}' data.csv  # CSV 按逗号分隔取列
awk 'NR==5' file.txt            # 只打印第5行
awk '{sum+=$1} END{print sum}' nums.txt   # 第1列求和
awk -F: '{print $1}' /etc/passwd   # 提取用户名

# sed — 流编辑
sed 's/old/new/g' file.txt      # 全局替换
sed -i 's/old/new/g' file.txt   # 直接修改文件
sed -n '10,20p' file.txt        # 打印第10-20行
sed '/^$/d' file.txt            # 删除空行
sed -i.bak 's/a/b/g' file.txt   # 修改前创建备份

# wc — 统计
wc -l file.txt                  # 行数
wc -w file.txt                  # 词数
wc -c file.txt                  # 字节数
ls *.py | wc -l                 # 统计文件数

# sort — 排序
sort file.txt                   # 默认排序
sort -rn numbers.txt            # 数值逆序（大→小）
sort -t',' -k3 -n data.csv     # CSV 按第3列数值排序
sort -u names.txt               # 排序 + 去重

# uniq — 去重（需先 sort）
sort file.txt | uniq            # 去重
sort file.txt | uniq -c         # 统计每行出现次数
sort file.txt | uniq -c | sort -rn  # 按频率排序（高频在前）
```

---

## 3. 进程管理

```bash
# ps — 查看进程
ps aux                           # 所有进程（最常用）
ps -ef                           # 显示父子进程关系
ps aux | grep nginx              # 查找特定进程
ps aux --sort=-%mem | head -10  # 内存占用前10

# top — 实时监控（q 退出）
top                              # 实时监控
top -c                           # 显示完整命令
top -u root                      # 只看指定用户

# htop — 增强版 top（需安装）
htop                             # 彩色交互式
htop -u root                     # 指定用户

# kill — 终止进程
kill 1234                        # 优雅终止（SIGTERM）
kill -9 1234                     # 强制杀死（SIGKILL）
killall nginx                    # 按进程名杀
pkill -f "runserver"             # 按命令模式杀

# lsof — 查看打开的文件/端口
lsof -i:8080                     # 查看占用8080端口的进程
lsof -i                          # 查看所有网络连接
lsof -u root                     # root用户打开的文件
lsof +D /var/log                 # 查看目录下被打开的文件
```

---

## 4. 网络

```bash
# curl — HTTP 请求
curl http://localhost:8080              # GET 请求
curl -X POST -d '{"key":"val"}' \        # POST JSON
     -H "Content-Type: application/json" \
     http://api.example.com/users
curl -o file.zip URL                    # 下载文件
curl -I https://example.com              # 只看响应头
curl -s -w "%{http_code}" URL            # 只输出状态码

# wget — 下载
wget https://example.com/file.tar.gz     # 下载文件
wget -c URL                              # 断点续传
wget -O newname.zip URL                  # 指定保存文件名
wget -q URL                              # 静默模式

# netstat — 网络连接（需 net-tools）
netstat -tlnp                            # 监听中的端口
netstat -an | grep ESTABLISHED            # 已建立的连接
netstat -tlnp | grep :8080               # 查看端口占用

# ss — netstat 替代（更快）
ss -tlnp                                 # 监听中的端口
ss -tlnp | grep :8080                    # 查看端口占用
ss -s                                    # 连接统计摘要

# ping — 测试连通性
ping -c 4 google.com                     # 发4个包后停止
ping -i 2 192.168.1.1                   # 每2秒一次
```

---

## 5. 权限管理

```bash
# chmod — 修改权限
chmod 755 script.sh              # rwxr-xr-x（所有者全权限，其他读+执行）
chmod 644 file.txt               # rw-r--r--（所有者读写，其他只读）
chmod +x script.sh               # 添加执行权限
chmod -R 755 mydir/              # 递归修改目录权限
chmod u+x,g-w file.txt          # 所有者加执行，组去掉写权限
```

> **数字含义**：`4=读(r)` `2=写(w)` `1=执行(x)`，如 `755 = rwxr-xr-x`

```bash
# chown — 修改所有者
chown user file.txt              # 改所有者
chown user:group file.txt        # 改所有者和组
chown -R user:group mydir/       # 递归修改整个目录

# sudo — 以管理员执行
sudo apt update                  # 以 root 权限执行
sudo -i                          # 进入 root 交互式 shell
sudo -u www-data cat /var/log/app.log   # 以指定用户执行
```

---

## 快速参考

| 场景 | 命令 |
|------|------|
| 找大文件 | `find . -type f -size +100M` |
| 杀占用端口的进程 | `lsof -i:8080` → `kill -9 PID` |
| 统计日志错误数 | `grep -c "error" app.log` |
| 找最频繁的IP | `sort access.log \| uniq -c \| sort -rn \| head` |
| 批量替换文本 | `sed -i 's/old/new/g' *.txt` |
| 查看端口占用 | `ss -tlnp \| grep :8080` |
| 递归改权限 | `chmod -R 755 mydir/` |
