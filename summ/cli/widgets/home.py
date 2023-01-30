import asyncio

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

from examples.otter.implementation.__main__ import summ_and_pipe
from summ.cli.widgets.output_tree import OutputTree


class Output(Static):
    final = reactive(False, init=False)
    text = reactive("")

    def watch_final(self) -> None:
        self.toggle_class("final")

    def watch_text(self) -> None:
        self.refresh(layout=True)

    def render(self):
        return self.text


class Home(Static):
    in_progress = reactive(False, init=False)

    def watch_in_progress(self, in_progress: bool) -> None:
        for button in self.query(Button):
            button.disabled = in_progress

    def on_output_tree_record_output(self, event: OutputTree.RecordOutput):
        self.log(event.text)
        self.query_one(Output).text = event.text
        self.scroll_to_widget(self.query_one(Output))

    def compose(self) -> ComposeResult:
        yield Vertical(
            Container(
                Button("Query", variant="success", id="query"),
                Button("Populate", variant="warning", id="populate"),
                id="buttons-container",
            ),
            Label("Progress", classes="heading"),
            OutputTree("What type of animal is Cronutt?", id="output-tree"),
            Label("Output", classes="heading"),
            Output(),
            id="container",
        )

    def action_query(self):
        summ, pipe = summ_and_pipe()
        return summ.query(
            "What type of animal is Cronutt?",
            n=2,
            classes=[],
            # corpus=list(pipe.corpus()),
            debug=True,
        )

    async def on_button_pressed(self, event: Button.Pressed):
        self.in_progress = True
        if event.button.id == "query":
            result = await asyncio.to_thread(self.action_query)
            self.query_one(Output).text = result
            self.query_one(Output).final = True
        self.in_progress = False
