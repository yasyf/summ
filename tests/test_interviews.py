from pathlib import Path

import pytest
import requests_mock

from user_interview_summary.classify.classes import Classes
from user_interview_summary.classify.classifier import (
    BaseClassifier,
    CompanyCategoryClassifier,
    DepartmentClassifier,
    IndustryClassifier,
    TitleClassifier,
)
from user_interview_summary.factify.factifier import Factifier
from user_interview_summary.pipeline import Pipeline
from user_interview_summary.splitter.splitter import Splitter


class TestInterviews:
    @pytest.fixture
    def interviews(self):
        return list((Path(__file__).parent.parent / "interviews").glob("*.txt"))

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
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def title_classifier(self):
        return TitleClassifier()

    @pytest.fixture
    def company_category_classifier(self):
        return CompanyCategoryClassifier()

    @pytest.fixture
    def department_classifier(self):
        return DepartmentClassifier()

    @pytest.fixture
    def industry_classifier(self):
        return IndustryClassifier()

    def test_splitter(self, interview: Path, splitter: Splitter):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        print(docs)
        assert len(docs) == 62

    def test_title_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        title_classifier: BaseClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = title_classifier.classify(docs[1])
        print(classes)
        assert Classes.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR is classes[0]

    def test_company_category_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        company_category_classifier: CompanyCategoryClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = company_category_classifier.classify(docs[1])
        print(classes)
        assert Classes.COMPANY_CATEGORY_RPA is classes[0]

    def test_department_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        department_classifier: DepartmentClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = department_classifier.classify(docs[1])
        print(classes)
        assert Classes.DEPARTMENT_RPA_DEVELOPMENT is classes[0]

    def test_industry_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        industry_classifier: IndustryClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = industry_classifier.classify(docs[1])
        print(classes)
        assert Classes.INDUSTRY_RPA_SOFTWARE is classes[0]

    def test_pipeline(self, interviews: list[Path], pipeline: Pipeline):
        pipe = pipeline.run([f.open(mode="r") for f in interviews])
        doc = next(pipe)
        assert doc.metadata["facts"] is not None

    def test_cache(
        self,
        interviews: list[Path],
        pipeline: Pipeline,
        requests_mock: requests_mock.Mocker,
    ):
        pipe = pipeline.run([f.open(mode="r") for f in interviews])
        requests_mock.stop()
        doc1 = next(pipe)
        doc2 = next(pipe)
        assert doc1.metadata["facts"] is not None
        assert doc2.metadata["facts"] != doc1.metadata["facts"]
        requests_mock.start()

        pipe = pipeline.run([f.open(mode="r") for f in interviews])
        doc1p = next(pipe)
        doc2p = next(pipe)
        assert doc1p.metadata["facts"] == doc1.metadata["facts"]
        assert doc2p.metadata["facts"] == doc2.metadata["facts"]
        assert doc2p.metadata["facts"] != doc1p.metadata["facts"]
