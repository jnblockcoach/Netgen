"""TUI tests (textual pilot): drive the app without a real TTY."""
import asyncio
import tempfile

from manager.tui import NetGenTui
from netgen import find_candidates
from netgen.generator import gen_folder


def _make_models(d, n=2):
    for i in range(n):
        c = find_candidates(5000, 20000, 1, seed=i + 1, arch_filter=['mlp'])[0]
        desc, code, params, inp, outp, mtype = c
        gen_folder(d, i + 1, desc, code, "M{}", params, inp, outp, mtype,
                   device_priority=['cpu'])


def test_tui_table_and_detail():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            table = app.query_one('#models-table')
            assert table.row_count == 2
            await pilot.press('down')
            await pilot.pause()
            # 光标移动即同步详情（RowHighlighted）
            assert '002' in app._detail_text and 'mlp' in app._detail_text
            app.action_detail()
            await pilot.pause()
            assert '002' in app._detail_text

    asyncio.run(_inner())


def test_tui_command_bar():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'info 1':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            assert '001' in app._detail_text

            for _ in range(6):
                await pilot.press('backspace')
            await pilot.pause()
            for ch in 'frobnicate':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            assert '未知指令' in '\n'.join(app._log_lines)

    asyncio.run(_inner())


def test_tui_generate_dialog_cancel():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            app.action_generate()
            await pilot.pause()
            await pilot.press('escape')
            await pilot.pause()
            assert len(app.screen_stack) == 1  # back to main screen

    asyncio.run(_inner())


def test_tui_train_subprocess_small_model():
    """Run a real 1-epoch training via the command bar (small model)."""
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d, n=1)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'train 001':
                await pilot.press(ch)
            await pilot.press('enter')
            # wait for the subprocess worker to finish
            for _ in range(600):
                await pilot.pause(0.05)
                if not app.busy and '完成' in '\n'.join(app._log_lines):
                    break
            log = '\n'.join(app._log_lines)
            assert '完成' in log and '退出码 0' in log
            # model should now be marked trained after auto-refresh
            table = app.query_one('#models-table')
            assert 'trained' in table.get_row_at(0)[4]

    asyncio.run(_inner())


def test_tui_focus_rules():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            # 启动焦点在表格（快捷键立即可用）
            assert app.focused is app.query_one('#models-table')
            # ctrl+e 聚焦命令栏
            await pilot.press('ctrl+e')
            await pilot.pause()
            assert app.focused is app.query_one('#cmd-input')
            # escape 回表格
            await pilot.press('escape')
            await pilot.pause()
            assert app.focused is app.query_one('#models-table')

    asyncio.run(_inner())


def test_tui_sort_command():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'sort params':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            assert app.sort_key == 'params'
            assert '已按 params 排序' in '\n'.join(app._log_lines)
            # 无效排序键
            for _ in range(12):
                await pilot.press('backspace')
            await pilot.pause()
            for ch in 'sort banana':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            assert '排序键无效' in '\n'.join(app._log_lines)

    asyncio.run(_inner())


def test_tui_refresh_keeps_selection():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press('down')
            await pilot.pause()
            app.action_detail()
            await pilot.pause()
            assert app._selected == '2'
            app.action_refresh()  # 刷新后选中/详情保持
            await pilot.pause()
            assert app._selected == '2'
            assert '002' in app._detail_text

    asyncio.run(_inner())


def test_tui_ps_and_deps_commands():
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'ps':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            log = '\n'.join(app._log_lines)
            assert '无运行中的训练进程' in log or 'pid' in log
            for _ in range(4):
                await pilot.press('backspace')
            await pilot.pause()
            for ch in 'deps':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            log = '\n'.join(app._log_lines)
            assert 'torch' in log and 'textual' in log

    asyncio.run(_inner())


def test_tui_params_dialog_save():
    """打开参数对话框 → 修改 → 仅保存 → config.py 更新。"""
    import os
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d, n=1)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 44)) as pilot:
            app.action_params()
            await pilot.pause()
            # 修改学习率字段
            await pilot.click('#p-lr')
            await pilot.pause()
            for _ in range(5):
                await pilot.press('backspace')
            await pilot.pause()
            for ch in '0.05':
                await pilot.press(ch)
            await pilot.pause()
            await pilot.click('#p-save')
            await pilot.pause()
            from netgen.manager import read_config
            import glob
            cfg = glob.glob(os.path.join(d, '001-*', 'config.py'))[0]
            c = read_config(cfg)
            assert c['LR'] == 0.05, c
            assert '已更新' in '\n'.join(app._log_lines)

    asyncio.run(_inner())


def test_tui_cfg_command():
    import os
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d, n=1)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'cfg 1':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            log = '\n'.join(app._log_lines)
            assert 'EPOCHS' in log and 'LR' in log and '训练参数' in log

    asyncio.run(_inner())


def test_tui_params_command_line_set():
    """命令栏: params 1 EPOCHS=30 直接写回 config.py。"""
    import os
    d = tempfile.mkdtemp(prefix='ngtui_')
    _make_models(d, n=1)
    app = NetGenTui(models_dir=d)

    async def _inner():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click('#cmd-input')
            await pilot.pause()
            for ch in 'params 1 EPOCHS=30':
                await pilot.press(ch)
            await pilot.press('enter')
            await pilot.pause()
            from netgen.manager import read_config
            import glob
            cfg = glob.glob(os.path.join(d, '001-*', 'config.py'))[0]
            assert read_config(cfg)['EPOCHS'] == 30
            assert 'updated' in '\n'.join(app._log_lines)

    asyncio.run(_inner())
