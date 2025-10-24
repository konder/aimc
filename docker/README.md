# AIMC MineDojo Docker 镜像构建指南

## 问题说明

在企业网络环境中（特别是阿里云环境），Docker 构建可能会遇到网络限制：
- 阿里云安全策略拦截外部网站访问
- 防火墙阻止 80/443 端口连接
- SSL证书验证失败

## 解决方案

### 方案 1: 配置网络代理（推荐）

如果你的环境有HTTP代理，在构建时设置代理：

```bash
docker build --platform linux/amd64 \
  --build-arg HTTP_PROXY=http://your-proxy:port \
  --build-arg HTTPS_PROXY=http://your-proxy:port \
  -t aimc-minedojo:latest .
```

### 方案 2: 手动下载 Miniconda（适合网络受限环境）

1. 在有网络访问的机器上下载 Miniconda：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 或者从国内镜像下载
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. 将下载的文件放到 `docker/` 目录下

3. 修改 Dockerfile：
   - 注释掉 `wget` 行
   - 取消 `COPY` 行的注释

4. 然后构建：
```bash
cd /Users/nanzhang/aimc/docker
docker build --platform linux/amd64 -t aimc-minedojo:latest .
```

### 方案 3: 申请网络白名单

根据错误消息，你需要：
1. 打开"云壳-防护记录-域名拦截"
2. 找到被拦截的域名（如 repo.anaconda.com）
3. 申请加白或报备后访问

需要加白的域名：
- repo.anaconda.com
- mirrors.tuna.tsinghua.edu.cn
- pypi.tuna.tsinghua.edu.cn
- deb.debian.org
- mirrors.aliyun.com

### 方案 4: 使用已构建的镜像（最简单）

如果其他地方已经有构建好的镜像，直接拉取：
```bash
# 从Docker Hub或其他镜像仓库拉取
docker pull <registry>/aimc-minedojo:latest
docker tag <registry>/aimc-minedojo:latest aimc-minedojo:latest
```

## 构建命令

在网络畅通的情况下，使用以下命令构建：

```bash
cd /Users/nanzhang/aimc/docker
docker build --platform linux/amd64 -t aimc-minedojo:latest .
```

## 运行镜像

构建完成后，运行容器：

```bash
# 运行容器并挂载项目目录
docker run -it --rm \
  --platform linux/amd64 \
  -v /Users/nanzhang/aimc:/workspace \
  aimc-minedojo:latest

# 在容器中验证环境
conda info --envs
python -c "import minedojo; print('MineDojo installed successfully!')"
```

## 镜像内容

本镜像包含：
- Ubuntu 20.04 (x86_64)
- OpenJDK 8
- Miniconda3
- Python 3.9 (minedojo-x86 环境)
- MineDojo 及其依赖
- 开发工具：git, curl, wget, build-essential

## 故障排查

### 问题：网络连接被拒绝
```
Could not connect to deb.debian.org:80 - connect (111: Connection refused)
```

**解决**：使用方案1（代理）或方案2（手动下载）

### 问题：SSL证书验证失败
```
ERROR: cannot verify repo.anaconda.com's certificate
```

**解决**：Dockerfile 已经使用 `--no-check-certificate` 参数

### 问题：阿里云安全策略拦截
```
抱歉，您要访问的网站不在安全策略默认允许的范围内
```

**解决**：使用方案3（申请白名单）或方案2（手动下载）

## 联系支持

如果遇到问题，请检查：
1. Docker 是否正确安装
2. 网络连接是否正常
3. 是否有足够的磁盘空间（至少5GB）

