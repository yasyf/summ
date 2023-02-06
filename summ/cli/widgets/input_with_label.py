from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Label, Static


class InputWithLabel(Static):
    def __init__(self, required: bool = True, **kwargs):
        self.required = required
        self.kwargs = kwargs
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label(self.kwargs["name"])
        yield Input(**self.kwargs)

    def on_mount(self):
        if self.required and not self.kwargs.get("value"):
            self.add_class("required")

    def on_input_changed(self, event: Input.Changed):
        if self.required and event.value:
            self.remove_class("required")
        elif self.required:
            self.add_class("required")
