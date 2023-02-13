from typing import TYPE_CHECKING

from textual.screen import Screen as TScreen
from textual.widgets import Footer

if TYPE_CHECKING:
    from summ.cli.app import SummApp


class Screen(TScreen):
    app: "SummApp"
    TITLE: str

    @property
    def title(self) -> str:
        return self.TITLE

    def on_screen_resume(self) -> None:
        self.refresh_title()
        self.query_one(Footer)._focus_changed(self)

    def on_screen_suspend(self) -> None:
        self.app.sub_title = ""

    def refresh_title(self):
        self.app.sub_title = self.title
