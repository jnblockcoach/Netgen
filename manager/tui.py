#!/usr/bin/env python3
"""NetGen TUI — 图形化命令行管理器（textual）

在终端里以面板/表格/快捷键方式管理模型：
  - 左侧表格：全部模型（ID/架构/参数/数据集/状态/最佳指标）
  - 右侧详情：选中模型的配置摘要与训练日志
  - 底部命令栏：内置指令（train/eval/sweep/generate/benchmark/...）
  - 底部日志区：训练/操作实时输出

用法:
    python manager/tui.py [--dir <模型目录>]
    NETGEN_DIR=<目录> python manager/tui.py

快捷键: r 刷新 · Enter 详情 · g 生成 · t 训练 · e 评估 · s 超参搜索
        b 对比训练 · m 资源采样 · c 清理 · q 退出
"""
import asyncio
import os
import subprocess
import sys
from typing import List, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             Log, Static)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELS_DIR = os.environ.get("NETGEN_DIR", os.path.join(ROOT, "generated_models"))
PYTHON = sys.executable
RUNPY = os.path.join(ROOT, "run.py")

sys.path.insert(0, ROOT)
from netgen.manager import scan_models, ModelInfo  # noqa: E402


# ── 帮助信息 ────────────────────────────────────────────────────────────

HELP_TEXT = """\
NetGen TUI — 图形化命令行管理器

【快捷键】
  r / F5    刷新模型列表        Enter    查看选中模型详情
  g         生成模型             t       训练选中模型
  e         评估选中模型         s       超参搜索选中模型
  b         对比训练             m       资源占用采样(monitor --once)
  c         清理模型             q       退出

【命令栏指令】（在底部 > 输入，回车执行）
  list                      刷新列表
  train <id> [--epochs N]   训练（默认 1 epoch，避免误跑大模型）
  eval <id>                 评估
  sweep <id> [--lrs ...]    超参搜索
  generate --range 5K-50K --count 5 [--arch mlp,cnn] [--dataset iris]
  benchmark [--workers N]   一键对比训练
  monitor                   资源占用采样
  clean [--force]           清理模型
  export [--format md]      导出报告
  archs                     架构列表
  help                      本帮助
  quit / q                  退出

【环境】
  NETGEN_DIR  模型目录（默认: <项目根>/generated_models）
  NETGEN_PYTHON  Python 解释器
"""


# ── 生成对话框 ──────────────────────────────────────────────────────────

class GenerateScreen(ModalScreen):
    """收集生成参数：--range --count --arch --dataset --device"""

    BINDINGS = [("escape", "dismiss(None)", "取消")]

    def compose(self) -> ComposeResult:
        yield Static("生成模型", classes="dlg-title")
        yield Label("参数范围 (必填, 如 5K-50K):")
        yield Input(placeholder="5K-50K", id="dlg-range")
        yield Label("数量 (默认 5):")
        yield Input(placeholder="5", id="dlg-count")
        yield Label("架构 (默认全部, 逗号分隔):")
        yield Input(placeholder="mlp,cnn,lstm", id="dlg-arch")
        yield Label("数据集 (默认 syn):")
        yield Input(placeholder="syn / iris / mnist / cifar10", id="dlg-dataset")
        yield Label("设备优先级 (默认自动):")
        yield Input(placeholder="cuda,mps / cpu", id="dlg-device")
        yield Horizontal(Button("生成", variant="primary", id="dlg-ok"),
                         Button("取消", id="dlg-cancel"), classes="dlg-btns")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "dlg-range":
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dlg-ok":
            self._submit()
        else:
            self.dismiss(None)

    def _submit(self) -> None:
        args = []
        rng = self.query_one("#dlg-range", Input).value.strip()
        if not rng:
            self.notify("参数范围不能为空", severity="error")
            return
        args += ["--range", rng]
        count = self.query_one("#dlg-count", Input).value.strip()
        if count:
            args += ["--count", count]
        arch = self.query_one("#dlg-arch", Input).value.strip()
        if arch:
            args += ["--arch", arch]
        ds = self.query_one("#dlg-dataset", Input).value.strip()
        if ds:
            args += ["--dataset", ds]
        dev = self.query_one("#dlg-device", Input).value.strip()
        if dev:
            args += ["--device", dev]
        self.dismiss(args)


class ConfirmScreen(ModalScreen):
    """确认对话框：yes/no"""

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    BINDINGS = [("escape", "dismiss(False)", "取消")]

    def compose(self) -> ComposeResult:
        yield Static(self.message, classes="dlg-title")
        yield Horizontal(Button("是", variant="error", id="dlg-yes"),
                         Button("否", variant="primary", id="dlg-no"),
                         classes="dlg-btns")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "dlg-yes")


# ── 主应用 ───────────────────────────────────────────────────────────────

class NetGenTui(App):
    TITLE = "NetGen 管理器"
    CSS = """
    #models-table {
        height: 1fr;
        border: round $primary;
    }
    #detail-panel {
        width: 45%;
        border: round $secondary;
        padding: 0 1;
    }
    #log-panel {
        height: 8;
        border: round $accent;
        background: $surface;
    }
    #cmd-input {
        dock: bottom;
    }
    .dlg-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .dlg-btns {
        margin-top: 1;
        align-horizontal: right;
    }
    #stat-bar {
        height: 1;
        color: $text-muted;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "刷新"),
        Binding("enter", "detail", "详情"),
        Binding("g", "generate", "生成"),
        Binding("t", "train", "训练"),
        Binding("e", "eval", "评估"),
        Binding("s", "sweep", "超参搜索"),
        Binding("b", "benchmark", "对比训练"),
        Binding("m", "monitor", "资源采样"),
        Binding("c", "clean", "清理"),
        Binding("q", "quit", "退出"),
    ]

    def __init__(self, models_dir: str = MODELS_DIR):
        super().__init__()
        self.models_dir = models_dir
        self.models: List[ModelInfo] = []
        self.busy = False  # 有子进程在跑
        self._log_lines: List[str] = []
        self._detail_text = ""

    # ── 界面组装 ──

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="stat-bar")
        yield Horizontal(
            DataTable(id="models-table", cursor_type="row"),
            VerticalScroll(Static("", id="detail-panel", markup=False),
                           id="detail-scroll"),
        )
        yield Static("", id="log-panel", markup=False)
        yield Input(placeholder="指令: train 001 | generate --range 5K-50K --count 5 | help",
                    id="cmd-input")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.add_columns("ID", "架构", "参数", "数据集", "状态", "最佳指标",
                          "Tier")
        self.action_refresh()
        self.query_one("#cmd-input", Input).focus()

    # ── 数据 ──

    def action_refresh(self) -> None:
        """重新扫描模型目录并刷新表格。"""
        self.models = scan_models(self.models_dir)
        table = self.query_one("#models-table", DataTable)
        table.clear()
        trained = sum(1 for m in self.models if m.status == "trained")
        for m in self.models:
            metric = ""
            if m.best_metric_value is not None:
                metric = f"{m.best_metric_name}={m.best_metric_value:.4f}"
            table.add_row(str(m.index), m.architecture, f"{m.params:,}",
                          m.dataset, m.status, metric, m.tier,
                          key=str(m.index))
        total = len(self.models)
        self.query_one("#stat-bar", Static).update(
            f"  {self.models_dir}  |  {total} 个模型（{trained} 已训练）  |  "
            f"{'忙（任务运行中…）' if self.busy else '空闲'}"
        )

    def _selected_id(self) -> Optional[str]:
        table = self.query_one("#models-table", DataTable)
        row_key = table.cursor_row
        if row_key is None:
            return None
        return table.get_row_at(row_key)[0]

    def _find_model(self, ident: str) -> Optional[ModelInfo]:
        for m in self.models:
            if str(m.index) == ident or ident in m.folder_name:
                return m
        return None

    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        self._show_detail(event.row_key.value)

    def _show_detail(self, ident: str) -> None:
        m = self._find_model(ident)
        panel = self.query_one("#detail-panel", Static)
        if m is None:
            self._detail_text = ""
            panel.update("")
            return
        lines = [
            f"[b]{m.folder_name}[/b]",
            f"架构: {m.architecture}",
            f"参数: {m.params:,}",
            f"数据集: {m.dataset}",
            f"状态: {m.status}",
        ]
        if m.best_metric_value is not None:
            lines.append(f"最佳: {m.best_metric_name}={m.best_metric_value:.4f}")
        lines.append(f"best_model.pth: {'✓' if m.has_best else '—'}")
        lines.append(f"checkpoints: {'✓' if m.has_checkpoints else '—'}")
        lines.append("")
        # 训练日志尾部
        log_path = os.path.join(m.folder_path, "training_log.md")
        if os.path.exists(log_path):
            lines.append("[b]训练日志尾[/b]")
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    tail = f.readlines()[-12:]
                lines.extend(l.rstrip() for l in tail)
            except OSError:
                pass
        self._detail_text = "\n".join(lines)
        panel.update(self._detail_text)

    # ── 子进程执行（实时输出到日志区）────────────────────────────

    def _log(self, text: str) -> None:
        self._log_lines = (self._log_lines + text.splitlines())[-300:]
        self.query_one("#log-panel", Static).update("\n".join(self._log_lines))

    @work(exclusive=True, group="run")
    async def _run(self, cmd: List[str], cwd: Optional[str] = None,
                   on_done=None) -> None:
        """异步跑子进程，stdout/stderr 实时显示在日志区。"""
        self.busy = True
        self.action_refresh()
        self._log(f"$ {' '.join(cmd)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self._log(line.decode(errors="replace").rstrip())
            rc = await proc.wait()
        except Exception as e:
            self._log(f"✗ 执行失败: {e}")
            rc = 1
        self._log(f"✓ 完成（退出码 {rc}）")
        self.busy = False
        self.action_refresh()
        if on_done:
            on_done(rc)

    # ── 指令执行 ──

    def _train_id(self, ident: str, extra: List[str]) -> None:
        m = self._find_model(ident)
        if m is None:
            self._log(f"✗ 找不到模型: {ident}")
            return
        epochs = ["--epochs", "1"]  # 默认 1 epoch，避免误跑大模型
        if "--epochs" not in extra:
            extra = epochs + extra
        cmd = [PYTHON, os.path.join(m.folder_path, "train.py")] + extra
        self._run(cmd, cwd=m.folder_path)

    def action_train(self) -> None:
        ident = self._selected_id()
        if ident:
            self._train_id(ident, [])

    def action_eval(self) -> None:
        ident = self._selected_id()
        if ident is None:
            return
        m = self._find_model(ident)
        if m is None:
            return
        eval_py = os.path.join(m.folder_path, "eval.py")
        if not os.path.exists(eval_py):
            self._log(f"✗ {ident} 没有 eval.py")
            return
        self._run([PYTHON, eval_py], cwd=m.folder_path)

    def action_sweep(self) -> None:
        ident = self._selected_id()
        if ident is None:
            return
        m = self._find_model(ident)
        if m is None:
            return
        cmd = [PYTHON, os.path.join(m.folder_path, "train.py"), "--epochs", "1"]
        self._log(f"提示: 完整超参搜索请用命令栏: sweep {ident} --lrs 0.001,0.01 "
                  f"--batches 64,128")
        self._log(f"（快捷键仅跑 1 个 lr=0.001 batch=64 的参考训练）")
        self._run(cmd, cwd=m.folder_path)

    def action_generate(self) -> None:
        self.push_screen(GenerateScreen(), self._on_generate_done)

    def _on_generate_done(self, args: Optional[List[str]]) -> None:
        if not args:
            return
        cmd = [PYTHON, RUNPY, "generate", "-o", self.models_dir] + args
        self._run(cmd)

    def action_benchmark(self) -> None:
        self.push_screen(
            ConfirmScreen("对比训练：训练全部未训练模型？\n（可在命令栏用 "
                          "benchmark --workers N / --time-budget 调整）"),
            self._on_benchmark_done)

    def _on_benchmark_done(self, ok: bool) -> None:
        if ok:
            self._run([PYTHON, RUNPY, "benchmark", "--dir", self.models_dir])

    def action_monitor(self) -> None:
        self._run([PYTHON, RUNPY, "monitor", "--once", "--interval", "1"])

    def action_clean(self) -> None:
        self.push_screen(
            ConfirmScreen("清理模型？\n（默认 dry-run 只预览，需 --force 才会删除）"),
            self._on_clean_done)

    def _on_clean_done(self, ok: bool) -> None:
        if ok:
            self._run([PYTHON, RUNPY, "clean", "--dir", self.models_dir, "--force"])

    def action_detail(self) -> None:
        ident = self._selected_id()
        if ident:
            self._show_detail(ident)
            self._log(f"显示 {ident} 详情（右侧面板）")

    def action_quit(self) -> None:
        if self.busy:
            self.notify("有任务在运行，完成后退出更安全。再按一次 q 强制退出。",
                        severity="warning")
            return
        self.exit()

    # ── 命令栏 ──

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "cmd-input":
            return
        line = event.value.strip()
        event.input.value = ""
        if not line:
            return
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        try:
            self._exec_cmd(cmd, args)
        except Exception as e:
            self._log(f"✗ 指令执行失败: {e}")

    def _exec_cmd(self, cmd: str, args: List[str]) -> None:
        if cmd in ("quit", "exit", "q"):
            self.action_quit()
        elif cmd in ("help", "h", "?"):
            self._log(HELP_TEXT)
        elif cmd in ("list", "ls", "refresh"):
            self.action_refresh()
        elif cmd in ("archs",):
            self._run([PYTHON, RUNPY, "archs", "--list"])
        elif cmd in ("info", "show"):
            ident = args[0] if args else self._selected_id()
            if ident:
                self._show_detail(ident)
        elif cmd == "train":
            if not args:
                self._log("用法: train <id> [--epochs N] [--lr X] [--device ...]")
                return
            self._train_id(args[0], args[1:])
        elif cmd == "eval":
            ident = args[0] if args else self._selected_id()
            if ident:
                self._show_detail(ident)
                self._log(f"评估 {ident}…（eval.py 输出见上）")
                m = self._find_model(ident)
                if m:
                    self._run([PYTHON, os.path.join(m.folder_path, "eval.py")],
                              cwd=m.folder_path)
        elif cmd == "sweep":
            ident = args[0] if args else self._selected_id()
            if not ident:
                self._log("用法: sweep <id> [--epochs N] [--lrs ...] [--batches ...]")
                return
            m = self._find_model(ident)
            if m is None:
                self._log(f"✗ 找不到模型: {ident}")
                return
            self._run([PYTHON, RUNPY, "sweep", ident, "--dir", self.models_dir,
                       "--device", "cpu"] + args, cwd=None)
        elif cmd == "generate":
            if "--range" not in args:
                self._log("用法: generate --range 5K-50K --count 5 "
                          "[--arch ...] [--dataset ...] [--device ...]")
                return
            self._run([PYTHON, RUNPY, "generate", "-o", self.models_dir] + args)
        elif cmd in ("benchmark", "bm"):
            self._run([PYTHON, RUNPY, "benchmark", "--dir", self.models_dir]
                      + args)
        elif cmd in ("monitor", "top"):
            self._run([PYTHON, RUNPY, "monitor", "--once", "--interval", "1"]
                      + args)
        elif cmd in ("clean",):
            flags = [a for a in args if a.startswith("-")]
            rest = [a for a in args if not a.startswith("-")]
            if "--force" not in flags and "-f" not in flags:
                self._log("默认 dry-run（只预览）。加 --force 才会删除。")
            self._run([PYTHON, RUNPY, "clean", "--dir", self.models_dir] + args)
        elif cmd == "export":
            self._run([PYTHON, RUNPY, "export", "--dir", self.models_dir]
                      + args)
        elif cmd in ("compare", "cmp"):
            self._run([PYTHON, RUNPY, "compare", "--dir", self.models_dir]
                      + args)
        elif cmd == "deps":
            self._run([PYTHON, RUNPY, "archs", "--list"])
            self._log("deps: 用 Python CLI 检查: python -c \"import torch, psutil\"")
        else:
            self._log(f"未知指令: {cmd}（help 查看全部指令）")


# ── 入口 ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="NetGen TUI 管理器")
    ap.add_argument("--dir", default=MODELS_DIR, help="模型目录")
    ap.add_argument("--dir-env", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    NetGenTui(models_dir=args.dir).run()


if __name__ == "__main__":
    main()
