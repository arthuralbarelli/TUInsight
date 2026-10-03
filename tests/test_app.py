import base64
import io
import unittest
from pathlib import Path

from nbformat.v4 import new_output
from PIL import Image
from textual.widgets import Button, Input, Static, TextArea
from textual_image.widget import Image as TerminalImage

from nbvim.kernel import ExecutionResult, KernelExecutionError
from nbvim.model import CellModel, NotebookModel, format_duration

from tuinsight.app import Block, TuInsight
from tuinsight.providers import ProposedCell, ProviderAuthError


def _png() -> str:
    image = Image.new("RGB", (2, 2), (0, 128, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class FakeProvider:
    def __init__(
        self, cells: list[ProposedCell], error: Exception | None = None
    ) -> None:
        self.cells = cells
        self.error = error
        self.questions: list[str] = []

    def available(self) -> bool:
        return self.error is None

    def status(self) -> str:
        return "fake"

    async def propose(self, question: str) -> list[ProposedCell]:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return list(self.cells)


class FakeKernel:
    def __init__(
        self,
        result: ExecutionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or ExecutionResult([], 1, duration_s=1.2)
        self.error = error
        self.calls: list[str] = []
        self.seen_status: list[str] = []
        self.app: TuInsight | None = None

    async def execute(self, source: str) -> ExecutionResult:
        self.calls.append(source)
        if self.app is not None:
            code = next(
                block
                for block in self.app.query(Block)
                if block.model.cell_type == "code" and block.model.source == source
            )
            self.seen_status.append(str(code.query_one(".run-status", Static).render()))
        if self.error is not None:
            raise self.error
        return self.result

    async def shutdown(self) -> None:
        pass


def _app(
    provider: FakeProvider,
    kernel: FakeKernel,
    notebook: NotebookModel | None = None,
    path: Path | None = None,
) -> TuInsight:
    app = TuInsight(
        notebook=notebook or NotebookModel(cells=[]),
        path=path,
        provider=provider,
        kernel=kernel,
    )
    kernel.app = app
    return app


class QuestionFlowTests(unittest.IsolatedAsyncioTestCase):
    def _provider(self) -> FakeProvider:
        return FakeProvider(
            [
                ProposedCell("markdown", "A média é 1."),
                ProposedCell("code", "print(1)"),
            ]
        )

    def _kernel(self) -> FakeKernel:
        png = _png()
        return FakeKernel(
            ExecutionResult(
                [
                    new_output(
                        "execute_result",
                        data={"text/plain": "   a\n0  1\n"},
                        execution_count=1,
                    ),
                    new_output(
                        "display_data",
                        data={
                            "text/plain": "<Figure size 1x1>",
                            "image/png": png,
                        },
                    ),
                ],
                execution_count=1,
                duration_s=1.2,
            )
        )

    async def test_ask_runs_the_cell_and_shows_nbvim_duration(self) -> None:
        provider = self._provider()
        kernel = self._kernel()
        app = _app(provider, kernel)
        async with app.run_test(size=(110, 40)) as pilot:
            self.assertEqual(list(app.query(Block)), [])
            self.assertTrue(app.query_one("#question-box").display)
            question = app.query_one("#question", Input)
            question.focus()
            question.value = "media"
            await pilot.press("enter")
            await pilot.pause()

            blocks = list(app.query(Block))
            self.assertEqual(provider.questions, ["media"])
            self.assertEqual(
                [block.model.cell_type for block in blocks], ["markdown", "code"]
            )
            self.assertEqual(kernel.calls, ["print(1)"])
            self.assertEqual(kernel.seen_status, ["rodando…"])

            code = blocks[1]
            self.assertFalse(code.query_one(".block-source", TextArea).display)
            self.assertEqual(code.model.metadata["nbvim"]["duration_ms"], 1200)
            self.assertEqual(
                str(code.query_one(".run-status", Static).render()),
                f"rodou em {format_duration(1.2)}",
            )
            self.assertEqual(
                str(code.query_one(".run-status", Static).render()), "rodou em 1.2s"
            )
            rendered = " ".join(
                str(widget.render()) for widget in code.query(".block-text")
            )
            self.assertIn("a", rendered)
            self.assertNotIn("<Figure", rendered)
            self.assertEqual(len(list(code.query(TerminalImage))), 1)
            self.assertEqual(len(list(app.query("#command-bar"))), 0)
            self.assertNotIn("j", app.active_bindings)
            self.assertNotIn("k", app.active_bindings)
            self.assertNotIn("colon", app.active_bindings)

    async def test_show_code_edit_clears_duration(self) -> None:
        provider = self._provider()
        kernel = self._kernel()
        app = _app(provider, kernel)
        async with app.run_test(size=(110, 40)) as pilot:
            app.query_one("#question", Input).value = "media"
            await pilot.click("#ask")
            await pilot.pause()
            code = list(app.query(Block))[1]
            await pilot.click(code.query_one(".toggle-code", Button))
            await pilot.pause()
            source = code.query_one(".block-source", TextArea)
            self.assertTrue(source.display)
            self.assertEqual(
                str(code.query_one(".toggle-code", Button).label), "Hide code"
            )
            source.focus()
            await pilot.press("x")
            await pilot.pause()
            self.assertIsNone(code.model.duration_s())
            self.assertEqual(str(code.query_one(".run-status", Static).render()), "")
            self.assertIn("x", source.text)

    async def test_error_is_one_sentence_until_traceback_is_asked(self) -> None:
        provider = FakeProvider([ProposedCell("code", "raise ValueError('bad value')")])
        kernel = FakeKernel(
            ExecutionResult(
                [
                    new_output(
                        "error",
                        ename="ValueError",
                        evalue="bad value",
                        traceback=["Traceback (most recent call last):", "hidden line"],
                    )
                ],
                execution_count=1,
                duration_s=1.2,
            )
        )
        app = _app(provider, kernel)
        async with app.run_test(size=(110, 40)) as pilot:
            app.query_one("#question", Input).value = "quebra"
            await pilot.click("#ask")
            await pilot.pause()
            block = app.query_one(Block)
            self.assertEqual(
                str(block.query_one(".error-sentence", Static).render()),
                "ValueError: bad value",
            )
            traceback = block.query_one(".traceback", Static)
            self.assertFalse(traceback.display)
            self.assertNotIn(
                "hidden line", str(block.query_one(".error-sentence").render())
            )
            self.assertEqual(
                str(block.query_one(".run-status", Static).render()), "rodou em 1.2s"
            )
            await pilot.click(block.query_one(".show-traceback", Button))
            await pilot.pause()
            self.assertTrue(traceback.display)
            self.assertIn("hidden line", str(traceback.render()))

    async def test_kernel_error_stays_one_sentence(self) -> None:
        provider = FakeProvider([ProposedCell("code", "print(1)")])
        failure = KernelExecutionError("No active Python environment.")
        failure.duration_s = 1.2
        kernel = FakeKernel(error=failure)
        app = _app(provider, kernel)
        async with app.run_test(size=(110, 30)) as pilot:
            app.query_one("#question", Input).value = "media"
            await pilot.click("#ask")
            await pilot.pause()
            block = app.query_one(Block)
            self.assertEqual(
                str(block.query_one(".error-sentence", Static).render()),
                "No active Python environment.",
            )
            self.assertFalse(block.query_one(".traceback", Static).display)
            self.assertEqual(
                str(block.query_one(".run-status", Static).render()), "rodou em 1.2s"
            )

    async def test_add_text_and_add_code(self) -> None:
        cell = CellModel(source="print(1)")
        app = _app(FakeProvider([]), FakeKernel(), NotebookModel(cells=[cell]))
        async with app.run_test(size=(110, 30)) as pilot:
            await pilot.pause()
            await pilot.click(".add-text")
            await pilot.pause()
            await pilot.click(".add-code")
            await pilot.pause()
            blocks = list(app.query(Block))
            self.assertEqual(
                [block.model.cell_type for block in blocks],
                ["code", "code", "markdown"],
            )
            self.assertFalse(blocks[1].query_one(".block-source", TextArea).display)
            self.assertTrue(blocks[2].query_one(".block-source", TextArea).display)

    async def test_stored_duration_is_shown_without_running_again(self) -> None:
        cell = CellModel(source="print(1)")
        cell.record_duration_s(65)
        kernel = FakeKernel()
        app = _app(FakeProvider([]), kernel, NotebookModel(cells=[cell]))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            block = app.query_one(Block)
            self.assertEqual(kernel.calls, [])
            self.assertFalse(block.query_one(".block-source", TextArea).display)
            self.assertEqual(
                str(block.query_one(".run-status", Static).render()), "rodou em 1m 05s"
            )

    async def test_auth_error_does_not_invent_cells(self) -> None:
        provider = FakeProvider(
            [], error=ProviderAuthError("OpenAI needs OPENAI_API_KEY.")
        )
        app = _app(provider, FakeKernel())
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#question", Input).value = "media"
            await pilot.click("#ask")
            await pilot.pause()
            self.assertEqual(list(app.query(Block)), [])
            self.assertIn(
                "OPENAI_API_KEY",
                str(app.query_one("#provider-status", Static).render()),
            )

    async def test_duration_is_saved_in_the_notebook(self) -> None:
        import tempfile

        provider = FakeProvider([ProposedCell("code", "print(1)")])
        kernel = FakeKernel(ExecutionResult([], 1, duration_s=1.2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "answer.ipynb"
            app = _app(provider, kernel, path=path)
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#question", Input).value = "media"
                await pilot.click("#ask")
                await pilot.pause()
            loaded = NotebookModel.load(path)
        self.assertEqual(loaded.cells[0].metadata["nbvim"]["duration_ms"], 1200)


class SourceTests(unittest.TestCase):
    def test_package_does_not_time_cells_or_copy_nbvim(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "tuinsight"
        text = "\n".join(path.read_text() for path in root.rglob("*.py"))
        self.assertNotIn("perf_counter", text)
        self.assertNotIn("monotonic", text)
        self.assertNotIn("VimTextArea", text)
        self.assertNotIn("nbvim.main", text)
        self.assertNotIn("Binding(", text)
        names = {path.name for path in root.rglob("*.py")}
        self.assertNotIn("kernel.py", names)
        self.assertNotIn("vim_editor.py", names)


if __name__ == "__main__":
    unittest.main()
