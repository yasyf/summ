# Summ

Intelligent question-answering and search for user interviews, powered by GPT-3.

## Demo

<script id="asciicast-wMwNAXKzFZtpxo5hV0Rxj0GhW" src="https://asciinema.org/a/wMwNAXKzFZtpxo5hV0Rxj0GhW.js" async></script>

## How it works

Summ starts with a corpus of user interview transcripts. These can be in any text format, such as exports from [Otter.ai](https://otter.ai).

We flow these through a pipeline:
[Import][summ.importers.Importer] -> [Split][summ.splitter.Splitter] -> [Classify][summ.classify.Classifier] | [Factify][summ.factify.Factifier] | [Structure][summ.structure.Structurer] | [Summarize][summ.summarize.Summarizer] -> [Embed][summ.embed.Embedder] to create a model which can answer questions across your entire dataset. Vector embeddings are persisted to [Pinecone](https://www.pinecone.io/).

Finally, we enable flexible [Querying][summ.query.Querier], following a recursive question-answering scheme.

Check out this [blog post](#) for more details.

## Installation

```console
pip install summ
```

You'll need an instance of [Redis Stack](https://redis.io/docs/stack/get-started/install/) running. We've found that `brew install redis-stack/redis-stack/redis-stack` is the fastest way to get up and running.

You'll also need to set two environment variables: `OPENAI_API_KEY`, and `PINECONE_API_KEY`.

## Quickstart

This quickstart is taken straight from the otter.ai [`example`](https://github.com/yasyf/summ/tree/main/examples/otter).


### Setup

#### Tags

The class `MyClasses` in [`implementation/classes.py`](https://github.com/yasyf/summ/tree/main/examples/otter/implementation/classes.py) sets out one categories of tags: audio source.

```python
from enum import StrEnum, auto
from summ.classify import Classes


class MyClasses(Classes, StrEnum):
    # SOURCE
    SOURCE_PODCAST = auto()
    SOURCE_INTERVIEW = auto()
    SOURCE_RADIO = auto()
```

#### Classifiers

The classifiers in [`implementation/classifier.py`](https://github.com/yasyf/summ/tree/main/examples/otter/implementation/classifier.py) use simple parameters to define a prompt for each category of tags. It is normally sufficient to simply provide [`CATEGORY`][summ.classify.Classifier.CATEGORY], [`VARS`][summ.classify.Classifier.VARS], and [`EXAMPLES`][summ.classify.Classifier.EXAMPLES]. You may also optionally specify a [`PREFIX`][summ.classify.Classifier.PREFIX] or [`SUFFIX`][summ.classify.Classifier.SUFFIX] for the prompt.

```python
from textwrap import dedent
from typing_extensions import override
from summ.classify.classifier import Classifier, Document
from .classes import MyClasses


class TypeClassifier(Classifier, classes=MyClasses):
    CATEGORY = "SOURCE"
    VARS = {
        "opening": "Opening Remarks",
        "source": "Audio Source",
    }
    EXAMPLES = [
        {
            "opening": "Welcome to radio one!",
            "source": MyClasses.SOURCE_RADIO,
        },
        {
            "opening": "This is the latest episode of the Science podcast.",
            "source": MyClasses.SOURCE_PODCAST,
        },
        {
            "opening": "We're sitting down today with Ben.",
            "source": MyClasses.SOURCE_INTERVIEW,
        },
    ]
    SUFFIX = f"If someone is being interviewd, the class is always {MyClasses.SOURCE_INTERVIEW}, even if the medium matches a different class."

    @override
    def classify(self, docs: list[Document]) -> dict[str, str]:
        return {"opening": docs[0].page_content}
```

#### CLI

Finally, in [`implementation/__main__.py`](https://github.com/yasyf/summ/tree/main/examples/otter/implementation/__main__.py), we:

1. Ensure our classifiers are imported
2. Construct a [`Summ`][summ.Summ] object, passing a [`Path`][pathlib.Path] to our training data.
3. Construct a custom [`Pipeline`][summ.pipeline.Pipeline] object which specifies the otter.ai import format.
4. Pass these two to [`summ.CLI`][summ.cli.CLI], which creates a command line interface for us.

```python
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
```

### Usage

#### Populate

Now, to populate our model, we can do:

```console
$ python -m implementation populate
```

#### Query

And to query it:

```console
$ python -m implementation query "What kind of animal is Cronutt?"
Cronutt is a California sea lion, a species of marine mammal.
```
