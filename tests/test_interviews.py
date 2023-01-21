from pathlib import Path

import pytest

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
        print(doc)
        assert doc.metadata["facts"] is not None
