from pathlib import Path

import pytest

from user_interview_summary.classify.classifier import (
    BaseClassifier,
    DepartmentClassifier,
)
from user_interview_summary.factify.factifier import Factifier
from user_interview_summary.splitter.splitter import Splitter


class TestInterviews:
    @pytest.fixture
    def interviews(self):
        return (Path(__file__).parent.parent / "interviews").glob("*.txt")

    @pytest.fixture
    def interview(self, interviews):
        return next(interviews)

    @pytest.fixture
    def splitter(self):
        return Splitter()

    @pytest.fixture
    def factifier(self):
        return Factifier()

    @pytest.fixture
    def classifier(self):
        return DepartmentClassifier()

    def test_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        classifier: BaseClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = classifier.classify(docs[1])
        print(classes)
        assert True

    def test_one_interview(
        self,
        interview: Path,
        splitter: Splitter,
        factifier: Factifier,
        classifier: BaseClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        facts = factifier.factify(docs[1])
        classes = classifier.classify(docs[1])
        print(facts, classes)
        assert True
