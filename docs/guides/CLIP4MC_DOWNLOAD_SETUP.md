# CLIP4MC 视频下载环境配置

生产机下载 YouTube 视频所需的增量软件清单。

## 系统依赖 (apt-get)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# 验证安装
ffmpeg -version
```

## Python 依赖 (pip)

### 最小依赖（仅下载视频）

```bash
pip install yt-dlp tqdm requests
```

### 完整依赖（下载 + 预处理）

```bash
pip install yt-dlp tqdm requests numpy opencv-python open-clip-torch
```

或者使用 transformers：

```bash
pip install yt-dlp tqdm requests numpy opencv-python transformers
```

## 依赖说明

| 包名 | 用途 | 是否必需 |
|------|------|----------|
| `yt-dlp` | YouTube 视频下载 | ✅ 下载必需 |
| `tqdm` | 进度条显示 | ⚠️ 可选（有 fallback） |
| `requests` | 下载数据集 JSON | ✅ 下载必需 |
| `numpy` | 数组处理 | ✅ 预处理必需 |
| `opencv-python` | 视频帧提取 | ✅ 预处理必需 |
| `open-clip-torch` | CLIP tokenizer | ✅ 预处理必需（二选一） |
| `transformers` | CLIP tokenizer | ✅ 预处理必需（二选一） |

## 快速安装命令

```bash
# 一键安装所有依赖
sudo apt-get update && sudo apt-get install -y ffmpeg
pip install yt-dlp tqdm requests numpy opencv-python open-clip-torch
```

## Cookies 配置

1. 在浏览器中登录 YouTube
2. 使用浏览器扩展导出 cookies（推荐 "Get cookies.txt LOCALLY"）
3. 保存到 `data/www.youtube.com_cookies.txt`

## 使用方式

```bash
# 下载测试集（约 2000 个样本）
./scripts/run_download_clip4mc.sh download

# 查看下载状态
./scripts/run_download_clip4mc.sh status

# 预处理为训练格式
./scripts/run_download_clip4mc.sh preprocess
```

## 常见问题

### 403 Forbidden 错误

1. 刷新浏览器 cookies
2. 重新导出 cookies 文件
3. 继续下载（会从断点恢复）

### ffmpeg 找不到

```bash
# 检查是否安装
which ffmpeg

# 如果未安装
sudo apt-get install -y ffmpeg
```

### yt-dlp 版本过旧

```bash
pip install --upgrade yt-dlp
```

