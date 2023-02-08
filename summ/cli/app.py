from typing import Any, cast

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from summ.cli.screens.main import MainScreen
from summ.cli.screens.settings import Settings, SettingsScreen
from summ.cli.screens.welcome import WelcomeScreen
from summ.pipeline import Pipeline
from summ.summ import Summ


class SummApp(App):
    TITLE = "Summ"
    SCREENS = {
        "main": MainScreen(),
        "settings": SettingsScreen(),
        "welcome": WelcomeScreen(),
    }
    CSS_PATH = "styles.css"

    def __init__(
        self,
        summ: Summ,
        pipe: Pipeline,
        is_demo: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.is_demo = is_demo
        self.summ = summ
        self.pipe = pipe

    @property
    def settings(self) -> Settings:
        return cast(SettingsScreen, self.get_screen("settings")).settings

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("main")
        if self.is_demo:
            self.push_screen("welcome")
        if not self.settings.valid():
            self.push_screen("settings")
