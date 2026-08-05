# NetGen 管理器（bash）

独立的 bash 管理器，内置指令，用于管理 NetGen 生成的模型。
与 Python CLI 完全分离（独立文件夹，不影响 `netgen` 包本身）。

## 快速开始

```bash
./manager/netgen.sh help                # 查看全部指令
./manager/netgen.sh deps                # 检查依赖
./manager/netgen.sh generate --range 5K-50K --count 5 --device cpu
./manager/netgen.sh list
./manager/netgen.sh train 001
./manager/netgen.sh logs 001
./manager/netgen.sh benchmark --workers 4
./manager/netgen.sh monitor
```

## 内置指令

| 指令 | 别名 | 说明 |
|------|------|------|
| `help` | `h` | 帮助 |
| `list` | `ls` | 列出模型 + 统计 |
| `info <id>` | `show` | 模型详情 |
| `compare` | `cmp` | 模型对比排行 |
| `generate` | `gen` | 批量生成（参数透传） |
| `train <id>` | | 训练单模型（参数透传） |
| `train-all` | | 训练全部未训练模型 |
| `eval <id>` | | 评估单模型 |
| `sweep <id>` | | 超参搜索 |
| `benchmark` | `bm` | 一键对比训练 |
| `monitor` | `top` | 实时资源监视 |
| `clean` | | 清理模型 |
| `export` | | 导出报告 |
| `archs` | | 列出架构 |
| `ps` | | 查看运行中的训练进程（bash 原生） |
| `logs <id>` | `log` | 最近训练日志（bash 原生 tail） |
| `config <id>` | `cfg` | 查看模型配置（bash 原生 cat） |
| `deps` | | 依赖检查（bash 原生探测） |
| `version` | `-v` | 版本信息 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NETGEN_DIR` | `<项目根>/generated_models` | 模型目录 |
| `NETGEN_PYTHON` | 自动检测 `.venv` → `venv` → `python3` | Python 解释器 |

```bash
NETGEN_DIR=/data/models ./manager/netgen.sh list
NETGEN_PYTHON=/opt/python/bin/python3 ./manager/netgen.sh deps
```

## bash 补全（可选）

```bash
source manager/completions/netgen.bash
```

启用后 `netgen.sh <TAB>` 补全指令名，`train <TAB>` 补全模型 ID，
`generate <TAB>` 补全选项。为脚本设置别名也可用：

```bash
alias ngm="$PWD/manager/netgen.sh"
complete -F _netgen_complete ngm
```

## 设计说明

- **透传指令**（generate/train/sweep/benchmark/monitor/...）：包装 `run.py`，
  额外参数原样传给 Python CLI，因此能力与 Python CLI 完全一致。
- **bash 原生指令**（ps/logs/config/deps）：不经过 Python，轻量快速。
- **不自动杀进程**：`ps` 只列出训练进程；结束进程需手动 `kill <pid>`。
