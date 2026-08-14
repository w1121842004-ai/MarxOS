# AutoDL 全集构建操作指南

在 AutoDL 租一张 GPU 完成全集（74 卷）corpus-v2.2 的构建，产物回传本机后做晋级验收。
本机只做「准备数据 + 上传」，构建与回传按本文档操作。

## 0. 总体流程

```text
本机：git push（已包含 v2.2 配置与脚本）
AutoDL：租卡 → 上传缓存 → clone → 跑脚本 → 回传产物
本机：校验 manifest → 切换 profile 晋级（旁路模式，旧索引不动）
```

## 1. 租卡（AutoDL 官网 www.autodl.com）

1. 注册 + 实名（几分钟）
2. **算力市场**选卡：**RTX 3090（24GB）** 或 4090——BGE-M3 用不到 A100，¥1.5–2/小时
3. 计费方式：**按量计费**（关机只收存储费，约 ¥0.5/天）
4. 镜像：选 **PyTorch 2.x** 基础镜像（带 python 3.10 + CUDA 12 的即可）
5. 地区：任意（就近）
6. 创建后进入「容器实例」页，记下三样：
   - SSH 登录指令（`ssh -p 端口 root@connect.xxx.autodl.com`）
   - 登录密码（控制台可见）
   - JupyterLab 地址（备用）

## 2. 上传缓存到 AutoDL（本机执行）

数据盘路径是 **`/root/autodl-tmp`**（系统盘关机清空，数据盘保留——所有数据放这里）。

```bash
# 在 AutoDL 上先建目录（SSH 上去执行一次）
ssh -p <端口> root@connect.<区域>.autodl.com "mkdir -p /root/autodl-tmp/MarxOS/data /root/autodl-tmp/MarxOS/rag"

# 本机上传（两个缓存目录 + rag 结构化数据，共约 1.3GB）
cd /Users/HONOR/Desktop/AIproject/MarxOS
scp -P <端口> -r data/ocr_cache_text_layer root@connect.<区域>.autodl.com:/root/autodl-tmp/MarxOS/data/
scp -P <端口> -r data/ocr_cache            root@connect.<区域>.autodl.com:/root/autodl-tmp/MarxOS/data/
scp -P <端口> -r rag/*.json               root@connect.<区域>.autodl.com:/root/autodl-tmp/MarxOS/rag/
```

> 注意：
> - **不需要传 PDF**（段落切分只读缓存；页码证据已在缓存 JSON 里）
> - 不需要传 `data/artifacts/`、`data/milvus_lite/`（旧产物不参与）
> - scp 上传约 10–30 分钟；也可以在 JupyterLab 网页里拖拽上传（更慢），或先传网盘再在实例里下载

## 3. AutoDL 实例上：拉代码 + 装环境

```bash
# SSH 登录后
cd /root/autodl-tmp
git clone https://github.com/w1121842004-ai/MarxOS.git
cd MarxOS
# 把上传的缓存挪进仓库目录
mv ../data/ocr_cache_text_layer data/
mv ../data/ocr_cache data/
mv ../rag/*.json rag/

# 环境（镜像自带 torch，装其余依赖即可）
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 镜像若已预装 torch 也可以直接用系统 python；`.venv` 方式最稳。
> 确认 GPU 可用：`python -c "import torch; print(torch.cuda.is_available())"` 输出 True。

## 4. 跑构建（一条命令）

```bash
cd /root/autodl-tmp/MarxOS
source .venv/bin/activate
bash scripts/autodl_build_v2_2.sh --cuda --batch 128
```

脚本依次执行：preflight → pages → paragraphs（含审计门禁）→ enrich → 子块切分 → **全量 embedding + BM25 + Milvus 写入**（checkpoint 续跑，中断重跑同命令即可）→ validate → manifest。

预计 4–7 小时（embedding 在 3090 上约 20–40 分钟，其余为 CPU 阶段）。进度看 stdout；中断后重跑 `bash scripts/autodl_build_v2_2.sh --cuda --batch 128` 会从 checkpoint 续建。

## 5. 回传产物（AutoDL → 本机）

构建完成后，在**本机**执行：

```bash
cd /Users/HONOR/Desktop/AIproject/MarxOS

# 1) 段落/富化/子块/BM25/manifest（约 1–1.5GB）
scp -P <端口> -r root@connect.<区域>.autodl.com:/root/autodl-tmp/MarxOS/data/artifacts/corpus_v2_2 data/artifacts/

# 2) Milvus 向量库（约 1–2GB）
scp -P <端口> -r root@connect.<区域>.autodl.com:/root/autodl-tmp/MarxOS/data/milvus_lite/marxos_corpus_v2_2.db data/milvus_lite/
```

> scp 下载速度取决于你本地带宽（一般 5–20MB/s，共约 15–40 分钟）。

## 6. 关机与计费

- 构建完成、产物回传验证无误后：**关机**（控制台点「关机」，只收存储费）或直接**释放实例**
- 释放前确认产物已回传；数据盘释放后不可恢复
- 本流程总花费预计：**¥10–20**（4–7 小时 × ¥1.5–2 + 上传下载不计费）

## 7. 本机晋级（回传后执行，见 rebuild_v2_runbook 的晋级流程）

```bash
# 校验 manifest 与 validate 报告
.venv/bin/python scripts/validate_index_manifest.py   # 需先更新 manifest 指向 v2.2
# 新旧检索对比、Web smoke、启动稳定性×10、回滚验收
# 全部通过后切换默认 profile → marxos_corpus_v2_2
```

## 常见坑

| 现象 | 处理 |
| --- | --- |
| scp 连不上 | AutoDL 控制台复制的是带端口的完整指令；密码在控制台「容器实例」页 |
| 磁盘空间不足 | 数据放 `/root/autodl-tmp`（数据盘）；`df -h` 确认剩余 >20GB |
| 构建到一半断开 | checkpoint 设计保证可续：重跑同一条 `autodl_build_v2_2.sh` 命令 |
| `torch.cuda.is_available()` 为 False | 镜像选错（CPU 镜像）；重开实例选 PyTorch 镜像 |
| 校验阶段不过 | 把 `data/artifacts/corpus_v2_2/audit_report.json` 与 validate 报告回传，按 issue code 处理（与 v2 流程相同） |
