#!/usr/bin/env bash
# ============================================================================
#  netgen.sh — NetGen 管理器（独立 bash 工具，内置指令）
#
#  用法:
#    ./manager/netgen.sh <指令> [参数...]
#
#  内置指令:
#    help         显示本帮助
#    list         列出全部模型（含统计）
#    info <id>    查看单个模型详情
#    compare      模型对比排行
#    generate     批量生成模型（参数透传 netgen generate）
#    train <id>   训练单个模型
#    train-all    训练全部未训练模型（= benchmark）
#    eval <id>    评估单个模型
#    sweep <id>   单模型超参搜索
#    benchmark    一键对比训练（并发/时间预算见 netgen benchmark --help）
#    monitor      实时监视 python 进程资源占用
#    clean        清理模型
#    export       导出对比报告
#    archs        列出全部架构
#    ps           查看正在运行的训练进程
#    logs <id>    查看模型最近训练日志
#    config <id>  查看模型配置
#    deps         检查运行依赖
#    version      显示版本
#
#  环境变量:
#    NETGEN_DIR    模型目录（默认: <项目根>/generated_models）
#    NETGEN_PYTHON Python 解释器（默认: 自动检测 venv -> python3）
# ============================================================================
set -uo pipefail

# ── 路径定位 ────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${NETGEN_DIR:-$ROOT/generated_models}"

# ── Python 解释器自动检测 ───────────────────────────────────────────────────
# 候选按优先级，取第一个“能导入 netgen”的解释器（避免选中无依赖的系统 python）
PYTHON="${NETGEN_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for cand in \
      "$ROOT/.venv/bin/python" \
      "$ROOT/venv/bin/python" \
      "$ROOT/.python/bin/python" \
      "$ROOT/../venv/bin/python" \
      python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c "import sys; sys.path.insert(0, '$ROOT'); import netgen" >/dev/null 2>&1; then
      PYTHON="$cand"
      break
    fi
  done
fi
[[ -n "$PYTHON" ]] || PYTHON=python3

# ── 颜色（非 TTY 自动禁用）─────────────────────────────────────────────────
if [[ -t 1 ]]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
  C_CYN=$'\033[36m'; C_BLD=$'\033[1m'; C_RST=$'\033[0m'
else
  C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""; C_BLD=""; C_RST=""
fi

die() { echo "${C_RED}错误: $*${C_RST}" >&2; exit 1; }
warn() { echo "${C_YEL}警告: $*${C_RST}" >&2; }
ok()   { echo "${C_GRN}✓ $*${C_RST}"; }

# 包装 Python CLI（run.py 在项目根）
ng() { "$PYTHON" "$ROOT/run.py" "$@"; }

# 解析模型 ID（数字或文件夹名）→ 文件夹路径
model_dir() {
  local id="$1"
  if [[ -d "$MODELS_DIR/$id" ]]; then
    echo "$MODELS_DIR/$id"; return 0
  fi
  if [[ "$id" =~ ^[0-9]+$ ]]; then
    local f
    for f in "$MODELS_DIR"/"$id"-*; do
      if [[ -d "$f" ]]; then echo "$f"; return 0; fi
    done
  fi
  local match
  match="$(find "$MODELS_DIR" -maxdepth 1 -type d -name "*${id}*" 2>/dev/null | head -1)"
  [[ -n "$match" ]] && { echo "$match"; return 0; }
  return 1
}

# ── 内置指令实现 ────────────────────────────────────────────────────────────

cmd_help() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "${C_BLD}当前设置:${C_RST}"
  echo "  项目根目录 : $ROOT"
  echo "  模型目录   : $MODELS_DIR"
  echo "  Python     : $PYTHON"
  echo
  echo "示例:"
  echo "  ./manager/netgen.sh generate --range 5K-50K --count 5 --device cpu"
  echo "  ./manager/netgen.sh benchmark --workers 4 --time-budget 30"
  echo "  ./manager/netgen.sh monitor --cpu 70 --gpu 80"
  echo "  ./manager/netgen.sh train 001"
  echo "  ./manager/netgen.sh logs 001"
}

cmd_list() {
  [[ -d "$MODELS_DIR" ]] || die "模型目录不存在: $MODELS_DIR（先运行 generate）"
  ng list --dir "$MODELS_DIR"
}

cmd_info() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh info <id>"
  ng info "$1" --dir "$MODELS_DIR"
}

cmd_compare() {
  ng compare --dir "$MODELS_DIR" "$@"
}

cmd_generate() {
  [[ $# -ge 2 ]] || die "用法: netgen.sh generate --range <lo>-<hi> [--count N] [--arch ...] [--dataset ...] [--device ...]"
  mkdir -p "$MODELS_DIR"
  ng generate -o "$MODELS_DIR" "$@"
}

cmd_train() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh train <id> [--epochs N] [--lr X] [--device ...]"
  local id="$1"; shift
  local dir; dir="$(model_dir "$id")" || die "找不到模型: $id"
  echo "${C_CYN}>> 训练 $id ($(basename "$dir"))${C_RST}"
  ng train "$id" --dir "$MODELS_DIR" "$@"
}

cmd_eval() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh eval <id>"
  local dir; dir="$(model_dir "$1")" || die "找不到模型: $1"
  ng eval "$1" --dir "$MODELS_DIR"
}

cmd_sweep() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh sweep <id> [--epochs N] [--lrs ...] [--batches ...]"
  ng sweep "$1" --dir "$MODELS_DIR" "${@:2}"
}

cmd_benchmark() {
  ng benchmark --dir "$MODELS_DIR" "$@"
}

cmd_train_all() {
  cmd_benchmark "$@"
}

cmd_monitor() {
  ng monitor "$@"
}

cmd_clean() {
  ng clean --dir "$MODELS_DIR" "$@"
}

cmd_export() {
  ng export --dir "$MODELS_DIR" "$@"
}

cmd_archs() {
  ng archs --list
}

cmd_ps() {
  echo "${C_BLD}运行中的训练/评估进程:${C_RST}"
  local found=0
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    found=1
    echo "$line"
  done < <(ps -eo pid,etime,%cpu,%mem,cmd 2>/dev/null | grep -E "train\.py|eval\.py|sweep\.py" | grep -v grep)
  [[ $found -eq 0 ]] && echo "  （无）"
  echo
  echo "提示: 可用 'kill <pid>' 手动结束（netgen 不会自动杀进程）"
}

cmd_logs() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh logs <id>"
  local dir; dir="$(model_dir "$1")" || die "找不到模型: $1"
  local log="$dir/training_log.md"
  [[ -f "$log" ]] || die "无训练日志: $log（先训练）"
  echo "${C_CYN}== $1 训练日志（最近 25 行）==${C_RST}"
  tail -n 25 "$log"
}

cmd_config() {
  [[ $# -ge 1 ]] || die "用法: netgen.sh config <id> [set KEY=VALUE ... | edit]"
  local id="$1"; shift
  local dir; dir="$(model_dir "$id")" || die "找不到模型: $id"
  local cfg="$dir/config.py"
  [[ -f "$cfg" ]] || die "无配置文件: $cfg"
  if [[ $# -eq 0 ]]; then
    cat "$cfg"
    return 0
  fi
  case "$1" in
    set)
      shift
      [[ $# -ge 1 ]] || die "用法: config <id> set EPOCHS=50 LR=0.01 ..."
      for a in "$@"; do
        [[ "$a" == *=* ]] || die "无效键值: $a（应为 KEY=VALUE）"
      done
      # 交给 netgen 的 set_model_params（保留注释格式）
      local err rc
      err="$("$PYTHON" - "$ROOT" "$MODELS_DIR" "$id" "$@" 2>&1 <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from netgen.manager import set_model_params
over = dict(k.split('=', 1) for k in sys.argv[4:])
print(set_model_params(sys.argv[2], sys.argv[3], **over))
PYEOF
)"
      rc=$?
      if [[ $rc -eq 0 ]]; then
        echo "$err"
      else
        die "参数更新失败: $err"
      fi
      ;;
    edit)
      ${EDITOR:-vi} "$cfg"
      echo "已保存 $cfg"
      ;;
    *)
      die "用法: config <id> [set KEY=VALUE ... | edit]"
      ;;
  esac
}

cmd_deps() {
  echo "${C_BLD}依赖检查:${C_RST}"
  printf "  %-28s" "项目根目录"
  echo "$ROOT"
  printf "  %-28s" "Python"
  "$PYTHON" --version 2>&1 || echo "✗ 未找到"
  printf "  %-28s" "PyTorch"
  "$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "✗ 未安装"
  printf "  %-28s" "psutil (monitor 需要)"
  "$PYTHON" -c "import psutil; print(psutil.__version__)" 2>/dev/null || echo "✗ 未安装 (pip install psutil)"
  printf "  %-28s" "scikit-learn (数据集)"
  "$PYTHON" -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "✗ 未安装（可选）"
  printf "  %-28s" "torchvision (图像数据集)"
  "$PYTHON" -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "✗ 未安装（可选，mnist/cifar10 需要）"
  printf "  %-28s" "nvidia-smi (GPU 监视)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "未检测到（无 GPU 或未装驱动）"
  printf "  %-28s" "netgen 包"
  "$PYTHON" -c "import netgen; print('OK')" 2>/dev/null || echo "✗ 无法导入（检查 PYTHONPATH=$ROOT）"
}

cmd_version() {
  local ver
  ver="$(cd "$ROOT" && git describe --tags --always 2>/dev/null || echo 'unknown')"
  echo "netgen manager $ver"
  echo "  项目根: $ROOT"
  echo "  模型目录: $MODELS_DIR"
}

# ── 指令分发 ────────────────────────────────────────────────────────────────

usage() { cmd_help; }

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in
  help|h|-h|--help)    cmd_help ;;
  list|ls)             cmd_list ;;
  info|show)           cmd_info "$@" ;;
  compare|cmp)         cmd_compare "$@" ;;
  generate|gen)        cmd_generate "$@" ;;
  train)               cmd_train "$@" ;;
  train-all|trainall)  cmd_train_all "$@" ;;
  eval)                cmd_eval "$@" ;;
  sweep)               cmd_sweep "$@" ;;
  benchmark|bm)        cmd_benchmark "$@" ;;
  monitor|top)         cmd_monitor "$@" ;;
  clean)               cmd_clean "$@" ;;
  export)              cmd_export "$@" ;;
  archs)               cmd_archs "$@" ;;
  ps)                  cmd_ps ;;
  logs|log)            cmd_logs "$@" ;;
  config|cfg)          cmd_config "$@" ;;
  deps)                cmd_deps ;;
  version|-v|--version) cmd_version ;;
  *)
    echo "${C_RED}未知指令: $cmd${C_RST}" >&2
    echo "可用指令: help list info compare generate train train-all eval sweep" >&2
    echo "          benchmark monitor clean export archs ps logs config deps version" >&2
    exit 1
    ;;
esac
