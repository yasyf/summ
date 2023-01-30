from typing import TYPE_CHECKING

from textual.screen import Screen as TScreen

if TYPE_CHECKING:
    from summ.cli.app import SummApp


class Screen(TScreen):
    app: "SummApp"
    TITLE: str

    @property
    def title(self) -> str:
        return self.TITLE

    def on_screen_resume(self) -> None:
        self.app.sub_title = self.title

    def on_screen_suspend(self) -> None:
        self.app.sub_title = ""
