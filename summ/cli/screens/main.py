import os
from enum import Enum

import pyperclip
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DirectoryTree, Footer, Header, Static

from summ.cli.screens.screen import Screen
from summ.cli.utils import push_screen
from summ.cli.widgets.file import File
from summ.cli.widgets.home import Home, Output
from summ.cli.widgets.reactive_tree import ReactiveTree


class ContentType(Enum):
    HOME = 0
    FILE = 1


class Content(Static):
    data: reactive[tuple] = reactive((ContentType.HOME,))

    def watch_data(self) -> None:
        self.query(None).remove()
        self.mount(self.widget())

    def render_home(self):
        return Home(self.app.summ, self.app.pipe)

    def render_file(self, path: str):
        return File(path)

    def widget(self):
        return getattr(self, f"render_{self.data[0].name.lower()}")(*self.data[1:])


class MainScreen(Screen):
    BINDINGS = [
        ("s", push_screen("settings"), "settings"),
        ("c", "copy", "copy output"),
        ("h,q,escape", "home", "home"),
    ]

    path = reactive(os.getcwd)

    def compute_path(self) -> str:
        return os.path.relpath(self.app.settings.corpus_path or os.getcwd())

    def on_screen_resume(self) -> None:
        self.path = self.compute_path()

    def watch_path(self, path: str) -> None:
        if tree := self.query(ReactiveTree):
            tree.first(ReactiveTree).path = path

    @property
    def title(self) -> str:
        return self.path

    def action_home(self) -> None:
        content = self.query_one(Content)
        content.data = (ContentType.HOME,)

    def action_copy(self) -> None:
        content = self.query_one(Content)
        if content.data[0] != ContentType.HOME:
            return
        pyperclip.copy(content.query_one(Home).query_one(Output).text)

    def on_directory_tree_file_selected(self, evt: DirectoryTree.FileSelected) -> None:
        content = self.query_one(Content)
        content.data = (ContentType.FILE, evt.path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            ReactiveTree(self.path, id="corpus-tree"),
            Vertical(Content(), id="content"),
            id="main-container",
        )
        yield Footer()
