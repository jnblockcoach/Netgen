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
            app.action_detail()
            await pilot.pause()
            assert '001' in app._detail_text and 'mlp' in app._detail_text

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
