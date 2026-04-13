# 服务器部署说明（详细版）

项目已在服务器 `/home/liwr/ai-safety` 目录，以下步骤完成「可通过域名/IP 访问」的部署。

---

## 第 0 步：确认虚拟环境路径（关键！）

Streamlit 和项目依赖装在虚拟环境里，systemd **必须使用虚拟环境的 Python**，否则报
`No module named streamlit`。

### 0.1 确认当前激活的 Python 是哪个

```bash
cd /home/liwr/ai-safety
which python3    # 若激活了 venv，会显示 venv 内路径
```

输出示例：
- 虚拟环境：`/home/liwr/ai-safety/venv/bin/python3`  ← 这个才对
- 系统 Python：`/usr/bin/python3`  ← 这个**不行**

### 0.2 列出 venv 目录

```bash
ls /home/liwr/ai-safety/venv/bin/
```

应该能看到 `python`、`python3`、`streamlit` 等文件。

若找不到 `venv`，可能是 `.venv`：

```bash
ls /home/liwr/ai-safety/.venv/bin/python
```

记下完整路径，下一步要用。

---

## 第 1 步：修改并安装 systemd 服务

### 1.1 修改 ExecStart 路径

```bash
sudo nano /etc/systemd/system/ai-safety.service
```

将文件内容替换为（把 `venv/bin/python` 改成你在第 0 步确认的路径）：

```ini
[Unit]
Description=AI Safety Governance Dashboard (Streamlit)
After=network.target

[Service]
User=liwr
WorkingDirectory=/home/liwr/ai-safety
EnvironmentFile=/home/liwr/ai-safety/.env
ExecStart=/home/liwr/ai-safety/venv/bin/python -m streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

保存：`Ctrl+O` → 回车 → `Ctrl+X`

### 1.2 重新加载并启动（每次改了 service 文件都要做）

```bash
sudo systemctl daemon-reload          # 让 systemd 重新读配置（必须！）
sudo systemctl enable ai-safety       # 开机自启
sudo systemctl restart ai-safety      # 启动/重启
```

### 1.3 检查是否正常运行

```bash
sudo systemctl status ai-safety
```

看到 `Active: active (running)` 才算成功。

若还是 `failed`，看日志（下面第 5 步）。

### 1.4 手动验证（可选，排错用）

不依赖 systemd，直接跑一次：

```bash
cd /home/liwr/ai-safety
source venv/bin/activate              # 激活 venv
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

这里如果报错，说明代码或配置有问题，不是 systemd 的问题。

---

## 第 2 步：配置演示密码（可选）

在 `.env` 文件末尾追加，演示时保护操作按钮不被领导误触：

```bash
echo "DEMO_PASSWORD=你设的密码" >> /home/liwr/ai-safety/.env
```

不设置则操作区默认对所有访问者开放。

---

## 第 3 步：配置 Nginx 反代

### 3.1 复制配置文件

```bash
sudo cp /home/liwr/ai-safety/deploy/nginx-ai-safety.conf /etc/nginx/sites-available/ai-safety
```

### 3.2 编辑域名（若有域名）

```bash
sudo nano /etc/nginx/sites-available/ai-safety
```

把 `server_name _;` 改为 `server_name 你的域名;`，没有域名就保持 `_`（匹配任意）。

### 3.3 启用配置

```bash
sudo ln -sf /etc/nginx/sites-available/ai-safety /etc/nginx/sites-enabled/ai-safety
```

若有 `default` 站点和你的配置冲突（都监听 80），可禁用：

```bash
sudo rm /etc/nginx/sites-enabled/default
```

### 3.4 检查并重载

```bash
sudo nginx -t                          # 检查语法
sudo systemctl reload nginx            # 重载（不中断现有连接）
```

### 3.5 访问测试

```bash
curl -I http://localhost               # 在服务器上测
```

返回 `200` 或 Streamlit 相关内容说明 Nginx 反代成功。

然后在你的电脑浏览器打开 `http://服务器IP`，看到看板页面即部署成功。

---

## 第 4 步：HTTPS（有域名时，强烈推荐）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d 你的域名
```

证书有效期 90 天，certbot 会自动续签，之后什么都不用做。

---

## 第 5 步：常见问题排查

### 查看详细错误日志

```bash
sudo journalctl -u ai-safety -n 100 --no-pager
```

### 常见报错与处理

| 报错 | 原因 | 解决方式 |
|------|------|----------|
| `No module named streamlit` | ExecStart 用的是系统 Python | 改 `ExecStart` 为 venv 路径，见第 1 步 |
| `No module named pydantic/pandas/...` | 同上，或 venv 里漏装 | `source venv/bin/activate && pip install -r requirements.txt` |
| `Permission denied: .env` | .env 权限不对 | `chmod 600 /home/liwr/ai-safety/.env` |
| `can't open file 'app.py'` | WorkingDirectory 写错 | 确认 `ai-safety.service` 里路径拼写正确 |
| `Address already in use: 8501` | 端口被占用（旧进程还在） | `sudo lsof -i:8501` 找到并 kill 掉旧进程 |
| Nginx 访问白屏 | 缺少 WebSocket 头 | 确认 nginx conf 里有 `Upgrade` 和 `Connection upgrade` 两行 |

### 更新代码后重启服务

```bash
cd /home/liwr/ai-safety
git pull                               # 若使用 git
sudo systemctl restart ai-safety
```

### 查看实时日志

```bash
sudo journalctl -u ai-safety -f        # Ctrl+C 退出
```

---

## 快速检查清单

部署前逐项打勾：

- [ ] `venv/bin/python -m streamlit --version` 能正常输出
- [ ] `.env` 文件存在且 `DASHSCOPE_API_KEY` 已填写
- [ ] `GUARDIAN_API_KEY` 已填写（用于卫报新闻同步）
- [ ] `ai-safety.service` 的 `ExecStart` 路径是 venv 内的 Python
- [ ] `sudo systemctl status ai-safety` 显示 `active (running)`
- [ ] `sudo nginx -t` 通过
- [ ] 浏览器能访问服务器 IP 并看到看板页面
