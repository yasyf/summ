from pathlib import Path

from summ import Pipeline, Summ
from summ.cli import CLI
from summ.splitter.otter import OtterSplitter

if __name__ == "__main__":
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

    CLI.run(summ, pipe)
