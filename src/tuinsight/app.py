"""Terminal notebook. A question first, then blocks with plain actions.

There is no vim: no j/k, no modes, and no command bar. Cell duration is the
one nbvim already stored. This app does not time the run again.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static, TextArea
from textual_image.widget import Image as TerminalImage

from nbvim.kernel import KernelExecutionError, ProjectKernel
from nbvim.model import CellModel, NotebookModel, format_duration

from .providers import (
    ProviderError,
    ProposedCell,
    select_provider,
)
from .render import error_sentence, image_from_output, output_traceback, plain_text


class Block(Vertical):
    """One notebook cell: result, duration, and the actions for that block."""

    def __init__(self, model: CellModel) -> None:
        super().__init__(classes="block")
        self.model = model
        self._ready = False
        self._text_editor = model.cell_type == "markdown" and model.source == ""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="actions"):
            yield Button("Run", classes="run")
            yield Button("Add text", classes="add-text")
            yield Button("Add code", classes="add-code")
            if self.model.cell_type == "code":
                yield Button("Show code", classes="toggle-code")
            traceback_button = Button("Show traceback", classes="show-traceback")
            traceback_button.display = False
            yield traceback_button
        if self.model.cell_type == "markdown":
            preview = Markdown(self.model.source, classes="block-markdown")
            preview.display = bool(self.model.source)
            yield preview
            editor = TextArea(self.model.source, language="markdown", soft_wrap=True)
            editor.add_class("block-source")
            editor.display = self._text_editor
            yield editor
        else:
            editor = TextArea(self.model.source, language="python", soft_wrap=True)
            editor.add_class("block-source")
            editor.display = False
            yield editor
        yield Vertical(classes="block-output")
        yield Static("", markup=False, classes="error-sentence")
        traceback_view = Static("", markup=False, classes="traceback")
        traceback_view.display = False
        yield traceback_view
        yield Static("", markup=False, classes="run-status")

    async def on_mount(self) -> None:
        await self._render_outputs()
        self._show_stored_duration()
        self._ready = True

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not event.text_area.has_class("block-source"):
            return
        if self._ready and self.model.set_source(event.text_area.text):
            self.query_one(".run-status", Static).update("")
        if self.model.cell_type == "markdown":
            preview = self.query_one(".block-markdown", Markdown)
            preview.update(self.model.source)
            preview.display = bool(self.model.source.strip())

    def toggle_code(self) -> None:
        source = self.query_one(".block-source", TextArea)
        button = self.query_one(".toggle-code", Button)
        source.display = not source.display
        button.label = "Hide code" if source.display else "Show code"
        if source.display:
            source.focus()

    def toggle_traceback(self) -> None:
        view = self.query_one(".traceback", Static)
        button = self.query_one(".show-traceback", Button)
        view.display = not view.display
        button.label = "Hide traceback" if view.display else "Show traceback"

    async def run(self) -> bool:
        """Run a code cell. Duration comes from the kernel result, not a second clock."""
        if self.model.cell_type != "code":
            return True
        status = self.query_one(".run-status", Static)
        status.update("rodando…")
        self.model.clear_duration()
        try:
            result = await self.app.kernel.execute(self.model.source)
        except KernelExecutionError as exc:
            self._show_failure(str(exc), "".join(traceback.format_exception(exc)))
            if exc.duration_s is not None:
                self.model.record_duration_s(exc.duration_s)
                self._show_stored_duration()
            self.app.save_notebook()
            return False
        self.model.outputs = list(result.outputs)
        self.model.execution_count = result.execution_count
        self.model.record_duration_s(result.duration_s)
        await self._render_outputs()
        self._show_stored_duration()
        self.app.save_notebook()
        return True

    def _show_stored_duration(self) -> None:
        seconds = self.model.duration_s()
        if seconds is None:
            return
        self.query_one(".run-status", Static).update(
            f"rodou em {format_duration(seconds)}"
        )

    def _show_failure(self, sentence: str, traceback_text: str) -> None:
        self.query_one(".error-sentence", Static).update(sentence.splitlines()[0])
        view = self.query_one(".traceback", Static)
        button = self.query_one(".show-traceback", Button)
        view.update(traceback_text)
        view.display = False
        button.display = True
        button.label = "Show traceback"

    async def _render_outputs(self) -> None:
        output = self.query_one(".block-output", Vertical)
        widgets = []
        sentences: list[str] = []
        traces: list[str] = []
        for item in self.model.outputs:
            if not isinstance(item, dict):
                continue
            if item.get("output_type") == "error":
                sentences.append(error_sentence(item))
                traces.append(output_traceback(item))
                continue
            image = image_from_output(item)
            if image is not None:
                widgets.append(TerminalImage(image, classes="block-image"))
                continue
            text = plain_text(item)
            if text:
                widgets.append(Static(text, markup=False, classes="block-text"))
        async with output.batch():
            await output.remove_children()
            if widgets:
                await output.mount(*widgets)
        error_view = self.query_one(".error-sentence", Static)
        traceback_view = self.query_one(".traceback", Static)
        traceback_button = self.query_one(".show-traceback", Button)
        if sentences:
            error_view.update(sentences[0])
            traceback_view.update("\n".join(traces))
            traceback_view.display = False
            traceback_button.display = True
            traceback_button.label = "Show traceback"
        else:
            error_view.update("")
            traceback_view.update("")
            traceback_view.display = False
            traceback_button.display = False


class TuInsight(App):
    """Question at the top. Blocks underneath. No vim keys."""

    CSS_PATH = Path(__file__).with_name("app.tcss")

    def __init__(
        self,
        notebook: NotebookModel | None = None,
        path: Path | None = None,
        provider=None,
        kernel=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.notebook = notebook or NotebookModel(cells=[])
        self.path = path
        self.provider = provider or select_provider()
        self.kernel = kernel or ProjectKernel(path or Path.cwd())
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="question-box"):
            yield Static("O que você quer saber?")
            yield Input(placeholder="Pergunte aqui", id="question")
            yield Button("Perguntar", id="ask")
            yield Static(self.provider.status(), markup=False, id="provider-status")
        with VerticalScroll(id="blocks"):
            for cell in self.notebook.cells:
                yield Block(cell)

    def on_mount(self) -> None:
        self.query_one("#question", Input).focus()

    async def on_unmount(self) -> None:
        shutdown = getattr(self.kernel, "shutdown", None)
        if shutdown is not None:
            await shutdown()

    def save_notebook(self) -> None:
        if self.path is not None:
            self.notebook.save(self.path)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question":
            await self._ask(event.value)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        if button.id == "ask":
            await self._ask(self.query_one("#question", Input).value)
            return
        block = button
        while block is not None and not isinstance(block, Block):
            block = block.parent
        if not isinstance(block, Block):
            return
        if button.has_class("run"):
            await self._run_block(block)
        elif button.has_class("add-text"):
            await self._add_block(block, "markdown")
        elif button.has_class("add-code"):
            await self._add_block(block, "code")
        elif button.has_class("toggle-code"):
            block.toggle_code()
        elif button.has_class("show-traceback"):
            block.toggle_traceback()

    async def _ask(self, question: str) -> None:
        status = self.query_one("#provider-status", Static)
        question = question.strip()
        if not question:
            status.update("Escreva o que você quer saber.")
            return
        if self._busy:
            status.update("Aguarde o bloco que está rodando.")
            return
        self._busy = True
        status.update("perguntando…")
        try:
            proposed = await self.provider.propose(question)
        except ProviderError as exc:
            status.update(str(exc))
            return
        finally:
            self._busy = False
        added = await self._mount_proposed(proposed)
        status.update(self.provider.status())
        for block in added:
            if block.model.cell_type != "code":
                continue
            if not await self._run_block(block):
                break

    async def _mount_proposed(self, proposed: list[ProposedCell]) -> list[Block]:
        container = self.query_one("#blocks", VerticalScroll)
        added: list[Block] = []
        for item in proposed:
            model = CellModel(cell_type=item.cell_type, source=item.source)
            self.notebook.add_cell(model)
            block = Block(model)
            await container.mount(block)
            added.append(block)
        self.save_notebook()
        return added

    async def _run_block(self, block: Block) -> bool:
        if self._busy:
            return False
        self._busy = True
        try:
            return await block.run()
        finally:
            self._busy = False

    async def _add_block(self, after: Block, cell_type: str) -> None:
        model = CellModel(cell_type=cell_type, source="")
        index = self.notebook.cells.index(after.model) + 1
        self.notebook.add_cell(model, index)
        block = Block(model)
        await self.query_one("#blocks", VerticalScroll).mount(block, after=after)
        self.save_notebook()
        if cell_type == "markdown":
            block.query_one(".block-source", TextArea).focus()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tuinsight")
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path("tuinsight.ipynb"),
        help="notebook to open or create",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "chatgpt", "bedrock"),
        help="which provider answers the question",
    )
    args = parser.parse_args(argv)
    if args.notebook.exists():
        notebook = NotebookModel.load(args.notebook)
    else:
        notebook = NotebookModel(cells=[])
        notebook.save(args.notebook)
    provider = select_provider(args.provider)
    TuInsight(notebook=notebook, path=args.notebook, provider=provider).run()


if __name__ == "__main__":
    main()
