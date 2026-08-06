#!/usr/bin/env python3
"""NetGen TUI — 图形化命令行管理器（textual）

终端面板式管理：
  - 左侧表格：全部模型（ID/架构/参数/数据集/状态/最佳指标/Tier）
  - 右侧详情：选中模型的配置摘要与训练日志尾部
  - 底部命令栏：内置指令（train/eval/sweep/generate/benchmark/compare/
    monitor/clean/export/archs/ps/deps/sort/help/quit）
  - 底部日志区：子进程实时输出

焦点规则（重要）：
  - 表格聚焦时，单键快捷键生效（r/g/t/e/s/b/m/c/q/Enter）
  - 命令栏聚焦时输入指令（Ctrl+E 聚焦命令栏，Esc 回到表格）
  - 启动默认聚焦表格

用法:
    python manager/tui.py [--dir <模型目录>]
    NETGEN_DIR=<目录> python manager/tui.py
"""
import asyncio
import os
import sys
from typing import List, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label,
                             Static)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELS_DIR = os.environ.get("NETGEN_DIR", os.path.join(ROOT, "generated_models"))
PYTHON = sys.executable
RUNPY = os.path.join(ROOT, "run.py")

sys.path.insert(0, ROOT)
from netgen.manager import (scan_models, read_config, update_config,
                            ModelInfo)  # noqa: E402


HELP_TEXT = """\
NetGen TUI — 图形化命令行管理器

【焦点】
  Ctrl+E   聚焦命令栏      Esc      回到表格
  表格聚焦 → 单键快捷键生效；命令栏聚焦 → 输入指令

【快捷键】（表格聚焦时）
  r / F5   刷新列表        Enter    查看选中模型详情
  g        生成（对话框）  t        训练选中（默认 1 epoch）
  e        评估选中        s        超参搜索选中（1 epoch）
  b        对比训练        m        资源占用采样
  c        清理（确认）    o        训练参数（写回 config.py）
  q        退出

【命令栏指令】
  list / ls                    刷新列表
  sort <key>                   按列排序: params|loss|val_acc|val_loss|acc
                                |index|params|dataset|status
  info <id>                    查看详情（右侧面板）
  params <id> [KEY=VALUE ...]  调整训练参数（表单或直接赋值，
                               如: params 001 EPOCHS=50 LR=0.01）
  cfg <id>                     查看当前 config.py
  train <id> [--epochs N] [--lr X] [--batch-size N] [--device ...] [--seed N]
  eval <id>                    评估
  sweep <id> [--epochs N] [--lrs ...] [--batches ...] [--device ...]
  generate --range 5K-50K --count 5
      [--arch mlp,cnn] [--preset cv] [--dataset iris|mnist|cifar10]
      [--device cuda,mps] [--seed N] [--jobs N]
  benchmark [--epochs N] [--workers N] [--force] [--time-budget MIN]
            [--device ...] [--retries N]
  train-all                    训练全部未训练模型（= benchmark）
  compare [--sort params|val_acc|val_loss] [--top N]
  monitor [--cpu 70] [--gpu 80] [--memory 60] [--interval 2] [--pid ...]
  clean [--force] [--keep-best N] [--untrained]   默认 dry-run 预览
  export [--format md|csv|json] [--output FILE]
  archs                        架构列表
  ps                           运行中的训练进程快照
  deps                         依赖探测
  help                         本帮助
  quit / q                     退出

【环境】
  NETGEN_DIR   模型目录（默认: <项目根>/generated_models）
  NETGEN_PYTHON Python 解释器
"""


# ── 对话框 ────────────────────────────────────────────────────────────

class TrainParamsScreen(ModalScreen):
    """训练参数调整：读 config.py 现值，可保存并训练/仅保存/取消。"""

    def __init__(self, folder_path: str, **kwargs):
        super().__init__(**kwargs)
        self.folder_path = folder_path
        self.cfg = read_config(os.path.join(folder_path, "config.py"))

    BINDINGS = [("escape", "dismiss(None)", "取消")]

    def compose(self) -> ComposeResult:
        c = self.cfg
        yield Static("训练参数（写回 config.py）", classes="dlg-title")
        yield Label(f"Epochs（当前 {c.get('EPOCHS', 30)}）:")
        yield Input(placeholder=str(c.get('EPOCHS', 30)), id="p-epochs")
        yield Label(f"学习率 LR（当前 {c.get('LR', 0.001)}）:")
        yield Input(placeholder=str(c.get('LR', 0.001)), id="p-lr")
        yield Label(f"Batch Size（当前 {c.get('BATCH_SIZE', 64)}）:")
        yield Input(placeholder=str(c.get('BATCH_SIZE', 64)), id="p-batch")
        yield Label(f"优化器（当前 {c.get('OPTIMIZER', 'adam')}; adam/sgd/adamw）:")
        yield Input(placeholder=str(c.get('OPTIMIZER', 'adam')), id="p-opt")
        yield Label(f"调度器（当前 {c.get('SCHEDULER', 'none')}; none/cosine/plateau/step）:")
        yield Input(placeholder=str(c.get('SCHEDULER', 'none')), id="p-sched")
        yield Label(f"Weight Decay（当前 {c.get('WEIGHT_DECAY', 0.0)}）:")
        yield Input(placeholder=str(c.get('WEIGHT_DECAY', 0.0)), id="p-wd")
        yield Label(f"Seed（当前 {c.get('SEED', 42)}）:")
        yield Input(placeholder=str(c.get('SEED', 42)), id="p-seed")
        yield Label(f"设备优先级（当前 {c.get('DEVICE_PRIORITY', ['cuda', 'mps'])}; 如 cuda,mps / cpu）:")
        yield Input(placeholder="cpu", id="p-device")
        yield Horizontal(
            Button("保存并训练", variant="primary", id="p-train"),
            Button("仅保存", id="p-save"),
            Button("取消", id="p-cancel"), classes="dlg-btns")

    def _overrides(self) -> dict:
        o = {}
        ints = {"EPOCHS": "p-epochs", "BATCH_SIZE": "p-batch", "SEED": "p-seed"}
        floats = {"LR": "p-lr", "WEIGHT_DECAY": "p-wd"}
        for key, wid in ints.items():
            v = self.query_one(f"#{wid}", Input).value.strip()
            if v:
                try:
                    o[key] = int(v)
                except ValueError:
                    self.notify(f"{key} 必须是整数", severity="error")
                    return None
        for key, wid in floats.items():
            v = self.query_one(f"#{wid}", Input).value.strip()
            if v:
                try:
                    o[key] = float(v)
                except ValueError:
                    self.notify(f"{key} 必须是数字", severity="error")
                    return None
        for key, wid in (("OPTIMIZER", "p-opt"), ("SCHEDULER", "p-sched")):
            v = self.query_one(f"#{wid}", Input).value.strip()
            if v:
                o[key] = v
        dev = self.query_one("#p-device", Input).value.strip()
        if dev:
            o["DEVICE_PRIORITY"] = [d.strip() for d in dev.split(",") if d.strip()]
        return o

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "p-cancel":
            self.dismiss(None)
            return
        o = self._overrides()
        if o is None:
            return
        if bid == "p-save":
            self.dismiss(("save", o))
        elif bid == "p-train":
            self.dismiss(("train", o))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        o = self._overrides()
        if o is not None:
            self.dismiss(("train", o))


class GenerateScreen(ModalScreen):
    """生成参数表单（常用项 + 高级项）。"""

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
        yield Label("预设 (可选: cv/nlp/gen/light/all):")
        yield Input(placeholder="cv", id="dlg-preset")
        yield Label("随机种子 (默认 42):")
        yield Input(placeholder="42", id="dlg-seed")
        yield Label("并行生成数 (默认 1):")
        yield Input(placeholder="1", id="dlg-jobs")
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
        for wid, flag in (("#dlg-count", "--count"), ("#dlg-arch", "--arch"),
                          ("#dlg-dataset", "--dataset"),
                          ("#dlg-device", "--device"),
                          ("#dlg-preset", "--preset"),
                          ("#dlg-seed", "--seed"),
                          ("#dlg-jobs", "--jobs")):
            val = self.query_one(wid, Input).value.strip()
            if val:
                args += [flag, val]
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


# ── 主应用 ─────────────────────────────────────────────────────────────

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
        Binding("o", "params", "训练参数"),
        Binding("ctrl+e", "focus_cmd", "命令栏"),
        Binding("escape", "focus_table", "表格"),
        Binding("q", "quit", "退出"),
    ]

    # 表格排序键 → (排序函数)
    SORT_KEYS = {
        "index": lambda m: m.index,
        "params": lambda m: m.params,
        "loss": lambda m: (m.best_metric_value if m.best_metric_name
                           in ("loss", "val_loss", "recon_loss", "g_loss")
                           else float("inf")),
        "acc": lambda m: -(m.best_metric_value if m.best_metric_name
                           in ("accuracy", "val_acc") else float("-inf")),
        "val_loss": lambda m: (m.best_metric_value if m.best_metric_name
                               == "val_loss" else float("inf")),
        "val_acc": lambda m: -(m.best_metric_value if m.best_metric_name
                               == "val_acc" else float("-inf")),
        "dataset": lambda m: m.dataset,
        "status": lambda m: m.status,
    }

    def __init__(self, models_dir: str = MODELS_DIR):
        super().__init__()
        self.models_dir = models_dir
        self.models: List[ModelInfo] = []
        self.busy = False  # 有子进程在跑
        self.sort_key = "index"
        self._log_lines: List[str] = []
        self._detail_text = ""
        self._selected = None  # 当前选中模型 id（刷新后保持）

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
        table.focus()  # 默认焦点在表格 → 快捷键立即可用

    # ── 焦点 ──

    def action_focus_cmd(self) -> None:
        self.query_one("#cmd-input", Input).focus()

    def action_focus_table(self) -> None:
        self.query_one("#models-table", DataTable).focus()

    # ── 数据 ──

    def action_refresh(self) -> None:
        """重新扫描并按当前排序键刷新表格，保持选中行。"""
        self.models = scan_models(self.models_dir)
        key = self.SORT_KEYS.get(self.sort_key, self.SORT_KEYS["index"])
        self.models.sort(key=key)
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
        # 保持选中：若之前有选中且仍存在，光标移回该行并刷新详情
        sel = self._selected
        if sel:
            sel_row = None
            for i, m in enumerate(self.models):
                if str(m.index) == sel or m.folder_name == sel:
                    sel_row = i
                    break
            if sel_row is None and len(sel) >= 3:
                for i, m in enumerate(self.models):
                    if sel in m.folder_name:
                        sel_row = i
                        break
            if sel_row is not None:
                try:
                    table.move_cursor(row=sel_row)
                except Exception:
                    pass
                self._show_detail(sel)
                if sel_row is not None:
                    pass
            elif self.models:
                try:
                    table.move_cursor(row=0)
                except Exception:
                    pass
                self._show_detail(str(self.models[0].index))
        elif self.models:
            try:
                table.move_cursor(row=0)
            except Exception:
                pass
            self._show_detail(str(self.models[0].index))
        total = len(self.models)
        self.query_one("#stat-bar", Static).update(
            f"  {self.models_dir}  |  {total} 个模型（{trained} 已训练）  |  "
            f"排序: {self.sort_key}  |  "
            f"{'忙（任务运行中…）' if self.busy else '空闲'}")

    def _selected_id(self) -> Optional[str]:
        table = self.query_one("#models-table", DataTable)
        try:
            row_key = table.cursor_row
        except Exception:
            return None
        if row_key is None:
            return None
        try:
            return table.get_row_at(row_key)[0]
        except Exception:
            return None

    def _find_model(self, ident: str) -> Optional[ModelInfo]:
        if not ident:
            return None
        for m in self.models:
            if str(m.index) == ident:
                return m
        for m in self.models:
            if m.folder_name == ident:
                return m
        if len(ident) >= 3:  # 模糊匹配仅限较长的关键词，避免 '1' 误命中
            for m in self.models:
                if ident in m.folder_name:
                    return m
        return None

    @on(DataTable.RowHighlighted)
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_detail(str(event.row_key.value))

    @on(DataTable.RowSelected)
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        self._show_detail(str(event.row_key.value))

    def _show_detail(self, ident: str) -> None:
        self._selected = ident  # 所有详情路径都同步选中状态
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
        self._log_lines = (self._log_lines + text.splitlines())[-400:]
        self.query_one("#log-panel", Static).update("\n".join(self._log_lines))

    @work(exclusive=True, group="run")
    async def _run(self, cmd: List[str], cwd: Optional[str] = None,
                   on_done=None) -> None:
        """异步跑子进程，stdout/stderr 实时显示在日志区。"""
        if self.busy:
            self._log(f"⏳ 有任务在运行，本任务已排队…")
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

    # ── 快捷键动作 ──

    def _train_id(self, ident: str, extra: List[str]) -> None:
        m = self._find_model(ident)
        if m is None:
            self._log(f"✗ 找不到模型: {ident}")
            return
        if "--epochs" not in extra:
            extra = ["--epochs", "1"] + extra  # 安全默认：1 epoch
        self._run([PYTHON, os.path.join(m.folder_path, "train.py")] + extra,
                  cwd=m.folder_path)

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
        """真正的超参搜索（安全默认：1 epoch × lr=0.001,0.01）。"""
        ident = self._selected_id()
        if ident is None:
            return
        m = self._find_model(ident)
        if m is None:
            return
        self._log(f"超参搜索 {ident}：1 epoch × lr=[0.001,0.01] batch=64 "
                  f"（命令栏可自定义: sweep {ident} --lrs ... --batches ...）")
        self._run([PYTHON, RUNPY, "sweep", ident, "--dir", self.models_dir,
                   "--epochs", "1", "--lrs", "0.001,0.01", "--batches", "64",
                   "--device", "cpu"])

    def action_params(self) -> None:
        """调整选中模型的训练参数（写回 config.py）。"""
        ident = self._selected_id()
        if ident is None:
            return
        m = self._find_model(ident)
        if m is None:
            return
        self.push_screen(TrainParamsScreen(m.folder_path),
                         lambda r: self._on_params_done(ident, r))

    def _on_params_done(self, ident: str, result) -> None:
        if not result:
            return
        mode, overrides = result
        m = self._find_model(ident)
        if m is None:
            return
        changed = update_config(os.path.join(m.folder_path, "config.py"),
                                **overrides)
        if changed:
            self._log(f"已更新 {ident} 参数: " +
                      ", ".join(f"{k}={v}" for k, v in changed.items()))
        if mode == "train":
            self._train_id(ident, [])

    def action_generate(self) -> None:
        self.push_screen(GenerateScreen(), self._on_generate_done)

    def _on_generate_done(self, args: Optional[List[str]]) -> None:
        if not args:
            return
        self._run([PYTHON, RUNPY, "generate", "-o", self.models_dir] + args)

    def action_benchmark(self) -> None:
        self.push_screen(
            ConfirmScreen("对比训练：训练全部未训练模型？\n（命令栏可加参: "
                          "benchmark --workers N --time-budget MIN）"),
            self._on_benchmark_done)

    def _on_benchmark_done(self, ok: bool) -> None:
        if ok:
            self._run([PYTHON, RUNPY, "benchmark", "--dir", self.models_dir])

    def action_monitor(self) -> None:
        self._run([PYTHON, RUNPY, "monitor", "--once", "--interval", "1"])

    def action_clean(self) -> None:
        self.push_screen(
            ConfirmScreen("清理模型？\n（默认 dry-run 只预览；命令栏加 "
                          "--force 才会删除）"),
            self._on_clean_done)

    def _on_clean_done(self, ok: bool) -> None:
        if ok:
            self._run([PYTHON, RUNPY, "clean", "--dir", self.models_dir,
                       "--force"])

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
        elif cmd == "sort":
            key = args[0] if args else "params"
            if key not in self.SORT_KEYS:
                self._log(f"✗ 排序键无效: {key}（可选: "
                          f"{', '.join(self.SORT_KEYS)}）")
                return
            self.sort_key = key
            self.action_refresh()
            self._log(f"已按 {key} 排序")
        elif cmd == "train":
            if not args:
                self._log("用法: train <id> [--epochs N] [--lr X] "
                          "[--batch-size N] [--device ...] [--seed N]")
                return
            self._train_id(args[0], args[1:])
        elif cmd in ("params", "options", "opts"):
            # params <id> 打开图形化参数调整；params <id> KEY=VALUE ... 直接设置
            ident = args[0] if args else self._selected_id()
            m = self._find_model(ident) if ident else None
            if m is None:
                self._log(f"用法: params <id> [KEY=VALUE ...] 或直接按 o 打开表单")
                return
            kv = [a for a in args[1:] if "=" in a]
            if kv:
                from netgen.manager import set_model_params
                self._log(set_model_params(self.models_dir, ident, **dict(
                    k.split("=", 1) for k in kv)))
            else:
                self.action_params()
        elif cmd == "eval":
            ident = args[0] if args else self._selected_id()
            m = self._find_model(ident) if ident else None
            if m is None:
                self._log(f"✗ 找不到模型: {ident or '(未选中)'}")
                return
            self._show_detail(ident)
            self._run([PYTHON, os.path.join(m.folder_path, "eval.py")],
                      cwd=m.folder_path)
        elif cmd == "sweep":
            ident = args[0] if args else self._selected_id()
            m = self._find_model(ident) if ident else None
            if m is None:
                self._log(f"用法: sweep <id> [--epochs N] [--lrs ...] "
                          f"[--batches ...] [--device ...]")
                return
            rest = [a for a in args[1:] if not a.startswith("--device")]
            dev = [a for a in args[1:] if a.startswith("--device")]
            self._run([PYTHON, RUNPY, "sweep", ident, "--dir",
                       self.models_dir] + dev + rest)
        elif cmd == "generate":
            if "--range" not in args:
                self._log("用法: generate --range 5K-50K --count 5 "
                          "[--arch ...] [--preset ...] [--dataset ...] "
                          "[--device ...] [--seed N] [--jobs N]")
                return
            self._run([PYTHON, RUNPY, "generate", "-o", self.models_dir]
                      + args)
        elif cmd in ("benchmark", "bm"):
            self._run([PYTHON, RUNPY, "benchmark", "--dir", self.models_dir]
                      + args)
        elif cmd == "train-all":
            self._run([PYTHON, RUNPY, "benchmark", "--dir", self.models_dir]
                      + args)
        elif cmd in ("compare", "cmp"):
            self._run([PYTHON, RUNPY, "compare", "--dir", self.models_dir]
                      + args)
        elif cmd in ("monitor", "top"):
            self._run([PYTHON, RUNPY, "monitor", "--once", "--interval", "1"]
                      + args)
        elif cmd == "clean":
            if "--force" not in args and "-f" not in args:
                self._log("默认 dry-run（只预览）。加 --force 才会删除。")
            self._run([PYTHON, RUNPY, "clean", "--dir", self.models_dir]
                      + args)
        elif cmd == "export":
            self._run([PYTHON, RUNPY, "export", "--dir", self.models_dir]
                      + args)
        elif cmd in ("cfg", "config"):
            ident = args[0] if args else self._selected_id()
            m = self._find_model(ident) if ident else None
            if m is None:
                self._log("用法: cfg <id> 查看当前 config.py")
                return
            cfg_path = os.path.join(m.folder_path, "config.py")
            if not os.path.exists(cfg_path):
                self._log(f"✗ {ident} 没有 config.py")
                return
            vals = read_config(cfg_path)
            shown = {k: vals[k] for k in
                     ("EPOCHS", "LR", "BATCH_SIZE", "OPTIMIZER",
                      "SCHEDULER", "WEIGHT_DECAY", "SEED",
                      "DEVICE_PRIORITY", "DATASET", "INPUT_DIM",
                      "OUTPUT_DIM") if k in vals}
            self._log(f"  {ident} 训练参数:")
            for k, v in shown.items():
                self._log(f"    {k} = {v}")
        elif cmd == "ps":
            self._cmd_ps()
        elif cmd == "deps":
            self._cmd_deps()
        else:
            self._log(f"未知指令: {cmd}（help 查看全部指令）")

    # ── 原生指令：ps / deps ──

    def _cmd_ps(self) -> None:
        """列出现在正在跑的训练/评估进程（不杀进程）。"""
        try:
            import psutil
        except ImportError:
            self._log("✗ 需要 psutil（pip install psutil）")
            return
        found = False
        for p in psutil.process_iter(["pid", "cmdline", "cpu_percent",
                                      "memory_percent"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
            except Exception:
                continue
            if any(t in cmd for t in ("train.py", "eval.py", "sweep.py",
                                      "benchmark", "run.py")):
                found = True
                self._log(f"  pid {p.info['pid']:>6d}  cpu {p.info['cpu_percent']:5.1f}%"
                          f"  mem {p.info['memory_percent']:4.1f}%  {cmd[-60:]}")
        if not found:
            self._log("  无运行中的训练进程")
        self._log("提示: 结束进程需手动 kill <pid>（管理器不杀进程）")

    def _cmd_deps(self) -> None:
        """探测关键依赖版本。"""
        import importlib
        for mod, hint in (("torch", ""), ("psutil", "（monitor 需要）"),
                          ("textual", "（TUI 需要）"),
                          ("sklearn", "（数据集，可选）"),
                          ("torchvision", "（mnist/cifar10，可选）")):
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "?")
                self._log(f"  {mod:<12s} {ver}  {hint}")
            except ImportError:
                self._log(f"  {mod:<12s} ✗ 未安装 {hint}")
        nv = os.path.exists("/usr/bin/nvidia-smi") or \
            os.path.exists("/usr/local/bin/nvidia-smi")
        self._log(f"  nvidia-smi   {'✓' if nv else '未检测到'}")


# ── 入口 ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="NetGen TUI 管理器")
    ap.add_argument("--dir", default=MODELS_DIR, help="模型目录")
    args = ap.parse_args()
    NetGenTui(models_dir=args.dir).run()


if __name__ == "__main__":
    main()
