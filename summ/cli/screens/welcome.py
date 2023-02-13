from textwrap import dedent

from rich.markdown import Markdown
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Static

from summ.cli.screens.screen import Screen


class WelcomeScreen(Screen):
    BINDINGS = [
        ("enter,esc,c", "pop_screen()", "continue"),
    ]
    MESSAGE = [
        dedent(
            """
        # Welcome to the summ demo!

        Before starting this demo, you'll need to have API keys for:
        - OpenAI
        - Pinecone

        It costs about $5 in OpenAI usage to do a one-time population
        of Pinecone with the included example data. If you'd like to see a video instead, go here: https://summ.readthedocs.io/en/stable/#demo.

        ## Example: Cronutt the Sea Otter

        This example dataset contains two podcasts and an interview
        with experts discussing the fate of Cronutt, a sea otter
        who suffered from epilepsy.

        ### Getting Started

        You'll see these two buttons on the next screen:
        """
        ),
        dedent(
            """
        One says "Populate", and you'll need to click this one first.
        This will use your Pinecone API key to populate a new index with the example dataset.
        This is necessary before running any queries.

        The other button says "Query", and this is what you can use to get summ to answer questions.
        Type your question into the field above the buttons and away you go!

        ### Try it out!

        Ask summ to tell you poor Cronutt's fate — was it a happy ending?
        """
        ),
    ]

    @property
    def title(self) -> str:
        return "Welcome!"

    def compose(self):
        yield Header()
        yield Vertical(
            Container(
                Static(Markdown(self.MESSAGE[0], justify="full"), classes="welcome-md"),
                Horizontal(
                    Button("Query", variant="success", disabled=True),
                    Button("Populate", variant="warning", disabled=True),
                    id="buttons-container",
                ),
                Static(Markdown(self.MESSAGE[1], justify="full"), classes="welcome-md"),
                Container(
                    Button("Continue", variant="success", id="continue"),
                    id="continue-container",
                ),
                id="welcome-container",
            ),
            id="welcome",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "continue":
            self.app.pop_screen()
