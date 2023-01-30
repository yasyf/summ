import os

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets._directory_tree import DirectoryTree


class ReactiveTree(Widget):
    path = reactive(os.getcwd)

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path

    def watch_path(self, path: str) -> None:
        self.query(DirectoryTree).remove()
        self.mount(DirectoryTree(path))
        self.query_one(DirectoryTree).show_root = False
