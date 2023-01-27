import sys
from pathlib import Path

import pytest
import requests_mock

from summ.factify.factifier import Factifier
from summ.pipeline import Pipeline
from summ.splitter.otter import OtterSplitter
from summ.splitter.splitter import Splitter

sys.path.append((Path(__file__).parent.parent / "examples" / "otter").as_posix())

from implementation.classes import MyClasses
from implementation.classifier import *


class TestInterviews:
    @pytest.fixture
    def interviews_path(self):
        return Path(__file__).parent.parent / "interviews"

    @pytest.fixture
    def interviews(self, interviews_path):
        print(interviews_path)
        return list(interviews_path.glob("*.txt"))

    @pytest.fixture
    def interview(self, interviews):
        return sorted(interviews)[2]

    @pytest.fixture
    def splitter(self):
        return OtterSplitter(
            speakers_to_exclude=[
                "Cindy Buckmaster",
                "Michelle Greenfield",
                "Vivica",
                "Deanna",
            ]
        )

    @pytest.fixture
    def factifier(self):
        return Factifier()

    @pytest.fixture
    def pipeline(self, interviews_path, splitter):
        pipe = Pipeline.default(interviews_path, "test")
        pipe.persist = False
        pipe.splitter = splitter
        return pipe

    @pytest.fixture
    def classifier(self):
        return TypeClassifier()

    def test_splitter(self, interview: Path, splitter: Splitter):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        assert len(docs) == 110

    def test_factifier(self, interview: Path, splitter: Splitter, factifier: Factifier):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)[:2]

        factifier.factify(docs[0])
        assert factifier.context != factifier.DEFAULT_CONTEXT

        facts = factifier.factify(docs[1])
        assert "is a veterinarian" in "\n".join(facts)

    def test_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        classifier: TypeClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)[:1]
        classes = classifier.run(docs)
        print(classes)
        assert MyClasses.SOURCE_PODCAST is classes[0]

    def test_pipeline(self, pipeline: Pipeline):
        pipe = pipeline.rung()
        doc = next(pipe)
        assert doc.metadata["facts"] is not None

    def test_cache(
        self,
        pipeline: Pipeline,
        requests_mock: requests_mock.Mocker,
    ):
        pipe = pipeline.rung()
        requests_mock.stop()
        doc1 = next(pipe)
        doc2 = next(pipe)
        assert doc1.metadata["facts"] is not None
        assert doc2.metadata["facts"] != doc1.metadata["facts"]
        requests_mock.start()

        pipeline.factifier = Factifier()
        pipe = pipeline.rung()
        doc1p = next(pipe)
        doc2p = next(pipe)
        assert doc1p.metadata["facts"] == doc1.metadata["facts"]
        assert doc2p.metadata["facts"] == doc2.metadata["facts"]
        assert doc2p.metadata["facts"] != doc1p.metadata["facts"]
