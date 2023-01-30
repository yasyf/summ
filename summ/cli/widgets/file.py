from rich.syntax import Syntax
from textual.reactive import reactive
from textual.widgets import Static


class File(Static):
    path = reactive("", init=False)

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path

    def watch_path(self, path: str) -> None:
        try:
            self.update(Syntax.from_path(path, line_numbers=True))
        except Exception as e:
            self.update(f"[bold red]{e}[/bold red]")
