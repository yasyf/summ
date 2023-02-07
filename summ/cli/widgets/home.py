import asyncio
import os

from rich.spinner import Spinner as RSpinner
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Static

from summ.cli.widgets.input_with_label import InputWithLabel
from summ.cli.widgets.output_tree import OutputTree
from summ.pipeline import Pipeline
from summ.summ import Summ


class Output(Static):
    final = reactive(False, init=False)
    text = reactive("")

    def watch_final(self) -> None:
        self.toggle_class("final")

    def watch_text(self) -> None:
        self.refresh(layout=True)

    def render(self):
        return self.text


class Spinner(Static):
    in_progress = reactive(False, init=False)

    def __init__(self, in_progress: bool, **kwargs) -> None:
        self.in_progress = in_progress
        super().__init__(**kwargs)

    def watch_in_progress(self) -> None:
        if self.in_progress:
            self.update(RSpinner("aesthetic"))
        else:
            self.update("")


class Home(Static):
    in_progress = reactive(False, init=False)
    question: reactive[str] = reactive("", init=False)

    def __init__(self, summ: Summ, pipe: Pipeline, **kwargs) -> None:
        self.summ = summ
        self.pipe = pipe
        super().__init__(**kwargs)

    def watch_in_progress(self, in_progress: bool) -> None:
        self.query_one(Spinner).in_progress = in_progress
        if not in_progress:
            return
        for button in self.query(Button):
            button.disabled = in_progress

    def on_output_tree_record_output(self, event: OutputTree.RecordOutput):
        self.log(event.text)
        self.query_one(Output).text = event.text
        self.scroll_to_widget(self.query_one(Output))

    def compose(self) -> ComposeResult:
        yield Vertical(
            InputWithLabel(
                name="Question",
                id="question",
                placeholder="What type of animal is Cronutt?",
            ),
            Container(
                Button("Query", variant="success", id="query", disabled=True),
                Button("Populate", variant="warning", id="populate"),
                id="buttons-container",
            ),
            Spinner(self.in_progress, id="spinner"),
            Label("Progress", classes="heading"),
            OutputTree(self.question, id="output-tree"),
            Label("Output", classes="heading"),
            Output(),
            id="container",
        )

    def action_query(self):
        return self.summ.query(
            self.question, classes=[], corpus=list(self.pipe.corpus()), debug=True
        )

    def action_populate(self):
        return self.summ.populate(self.pipe.importer.dir, pipe=self.pipe)

    async def on_button_pressed(self, event: Button.Pressed):
        output, tree = self.query_one(Output), self.query_one(OutputTree)
        self.in_progress = True

        if event.button.id == "query":
            result = await asyncio.to_thread(self.action_query)
            output.text = result
            tree.outputs[tree.last_node.data] = result
        elif event.button.id == "populate":
            tree.title = "Path"
            tree.question = os.path.relpath(self.app.settings.corpus_path)
            result = await asyncio.to_thread(self.action_populate)

        output.final = True
        event.button.disabled = False
        self.in_progress = False

    def on_input_changed(self, event: Input.Changed):
        self.question = event.value
        self.query_one(OutputTree).question = event.value
        if event.value:
            self.query_one("#query", Button).disabled = False
            self.query_one("#populate", Button).disabled = True
        else:
            self.query_one("#query", Button).disabled = True
            self.query_one("#populate", Button).disabled = False
