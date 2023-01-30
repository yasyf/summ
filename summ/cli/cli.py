import itertools
from pathlib import Path

import click
import langchain
from pydantic import BaseModel

from summ.classify import Classes, Classifier
from summ.pipeline import Pipeline
from summ.summ import Summ


class Options(BaseModel):
    debug: bool
    verbose: bool


class CLI:
    """Provides a convient way to serve a Summ CLI."""

    @staticmethod
    def run(summ: Summ, pipe: Pipeline):
        """Starts the CLI.

        Args:
            summ: The Summ instance to use. You should pre-populate it with the path to your data.
            pipe: The Pipeline instance to use. You can a specify custom [Splitter][summ.splitter.Splitter] for different sources.

        Example:
            ```python
            from pathlib import Path

            from summ import Pipeline, Summ
            from summ.cli import CLI
            from summ.splitter.otter import OtterSplitter

            from my.classifiers import *

            if __name__ == "__main__":
                summ = Summ(index="rpa-user-interviews")

                path = Path(__file__).parent.parent / "interviews"
                pipe = Pipeline.default(path, summ.index)
                pipe.splitter = OtterSplitter(speakers_to_exclude=["markie"])

                CLI.run(summ, pipe)
            ```
        """

        @click.group()
        @click.option("--debug/--no-debug", default=True)
        @click.option("--verbose/--no-verbose", default=False)
        @click.pass_context
        def cli(ctx, debug: bool, verbose: bool):
            ctx.obj = Options(debug=debug, verbose=verbose)
            langchain.verbose = verbose

        @cli.command()
        @click.pass_context
        def populate(ctx: click.Context):
            summ.populate(
                Path(pipe.importer.dir), pipe=pipe, parallel=not ctx.obj.verbose
            )

        class_options = set(
            itertools.chain.from_iterable(
                [list(c.classes) for c in Classifier.classifiers.values()]
            )
        )

        if not class_options:
            click.secho("Warning: No classifiers detected.", fg="yellow")

        @cli.command()
        @click.argument("query", nargs=1)
        @click.option("-n", default=3)
        @click.option(
            "--classes",
            multiple=True,
            default=[],
            type=click.Choice(list(class_options), case_sensitive=False),
        )
        @click.pass_context
        def query(ctx: click.Context, query: str, n: int, classes: list[Classes]):
            response = summ.query(
                query,
                n=n,
                classes=classes,
                corpus=list(pipe.corpus()),
                debug=ctx.obj.debug,
            )
            click.echo("\n")
            click.secho(response)

        cli()
