from typing import cast

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from summ.cli.screens.main import MainScreen
from summ.cli.screens.settings import Settings, SettingsScreen
from summ.cli.utils import push_screen


class SummApp(App):
    TITLE = "Summ"
    SCREENS = {"main": MainScreen(), "settings": SettingsScreen()}
    CSS_PATH = "styles.css"

    @property
    def settings(self) -> Settings:
        return cast(SettingsScreen, self.get_screen("settings")).settings

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("main")
        if not self.settings.valid():
            self.push_screen("settings")


if __name__ == "__main__":
    SummApp().run()
