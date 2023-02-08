from pathlib import Path

from summ import Pipeline, Summ
from summ.cli import CLI
from summ.splitter.otter import OtterSplitter

# Make sure to import our Classifiers:
from .classifier import *

# https://summ.readthedocs.io/en/stable/#cli


def summ_and_pipe():
    summ = Summ(index="cronutt-facts")

    path = Path(__file__).parent.parent / "interviews"
    pipe = Pipeline.default(path, summ.index)
    pipe.splitter = OtterSplitter(
        speakers_to_exclude=[
            "Cindy Buckmaster",
            "Michelle Greenfield",
            "Vivica",
            "Deanna",
        ]
    )

    return summ, pipe


def main():
    summ, pipe = summ_and_pipe()
    # Remove is_demo=True for your own app!
    CLI.run(summ, pipe, is_demo=True)


if __name__ == "__main__":
    main()
