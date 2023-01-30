import os
from pathlib import Path
from typing import Optional

from dotenv import set_key
from pydantic import BaseSettings
from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Label, Static

from summ.cli.screens.screen import Screen
from summ.cli.widgets.input_with_label import InputWithLabel

CONFIG_PATH = (
    Path(os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")))
    / "summ"
    / "env.sh"
)
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    openai_api_key: Optional[str]
    pinecone_api_key: Optional[str]
    corpus_path: Optional[str]

    class Config:
        env_file = CONFIG_PATH

    def valid(self) -> bool:
        return all(self.dict().values())

    def write(self):
        for key, value in self.dict().items():
            set_key(CONFIG_PATH, key, value)
            os.environ[key.upper()] = value


class SettingsScreen(Screen):
    TITLE = "Settings"
    settings: reactive[Settings] = reactive(Settings)
    valid = reactive(False)

    def compute_valid(self) -> bool:
        return self.settings.valid()

    def watch_valid(self, valid: bool):
        if container := self.query(Container):
            container.first().query_one("#save", Button).disabled = not valid

    def compose(self) -> ComposeResult:
        yield Header()

        yield InputWithLabel(
            name="OpenAI API Key",
            id="openai_api_key",
            placeholder="sk-...",
            value=self.settings.openai_api_key,
        )
        yield InputWithLabel(
            name="Pinecone API Key",
            id="pinecone_api_key",
            placeholder="...",
            value=self.settings.pinecone_api_key,
        )
        yield InputWithLabel(
            name="Data Directory",
            id="corpus_path",
            placeholder="/path/to/files",
            value=self.settings.corpus_path,
        )
        yield Container(
            Button(disabled=not self.valid, label="Save", id="save"),
            id="buttons-container",
        )

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save":
            self.settings.write()
        self.app.pop_screen()

    def on_input_changed(self, event: Input.Changed):
        if event.input.id:
            self.settings = self.settings.copy(update={event.input.id: event.value})
