# bash completion for manager/netgen.sh
#
# 启用（追加到 ~/.bashrc 或 source 一次）:
#   source <项目根>/manager/completions/netgen.bash
#
# 或为本脚本加别名后同样可用:
#   alias ngm=/path/to/netgen.sh
#   complete -F _netgen_complete ngm

_netgen_commands="help list info compare generate train train-all eval sweep benchmark monitor clean export archs ps logs config deps version"

_netgen_complete() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  local prev="${COMP_WORDS[COMP_CWORD-1]}"
  local cmd="${COMP_WORDS[1]}"
  local root models_dir

  root="$(cd "$(dirname "$(readlink -f "${COMP_WORDS[0]}" 2>/dev/null || echo "${COMP_WORDS[0]}")")/.." 2>/dev/null && pwd)"
  models_dir="${NETGEN_DIR:-$root/generated_models}"

  # 第一级：指令名
  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=($(compgen -W "$_netgen_commands" -- "$cur"))
    return
  fi

  # 模型 ID 补全（info/train/eval/sweep/logs/config 的第二参数）
  case "$cmd" in
    info|train|eval|sweep|logs|config)
      if [[ $COMP_CWORD -eq 2 ]]; then
        COMPREPLY=($(compgen -W "$(ls "$models_dir" 2>/dev/null \
          | sed -E 's/^([0-9]{3})-.*/\1/' | sort -u)" -- "$cur"))
        return
      fi
      ;;
  esac

  # generate/benchmark 的选项补全
  case "$cmd" in
    generate)
      COMPREPLY=($(compgen -W "--range --count --arch --preset --dataset --device --seed --jobs --output -o" -- "$cur"))
      return
      ;;
    benchmark)
      COMPREPLY=($(compgen -W "--epochs --lr --batch-size --workers --force --time-budget --device --dir --seed" -- "$cur"))
      return
      ;;
    monitor)
      COMPREPLY=($(compgen -W "--cpu --gpu --memory --interval --duration --once --pid" -- "$cur"))
      return
      ;;
  esac
}

complete -F _netgen_complete netgen.sh ngm ng
