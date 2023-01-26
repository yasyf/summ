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
        return Path(__file__).parent.parent / "examples" / "otter" / "interviews"

    @pytest.fixture
    def interviews(self, interviews_path):
        print(interviews_path)
        return list(interviews_path.glob("*.txt"))

    @pytest.fixture
    def interview(self, interviews):
        return sorted(interviews)[2]

    @pytest.fixture
    def splitter(self):
        return OtterSplitter(speakers_to_exclude=["markie"])

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
        title_classifier: TitleClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = title_classifier.run(docs[1])
        print(classes)
        assert MyClasses.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR is classes[0]

    def test_company_category_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        company_category_classifier: CompanyCategoryClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = company_category_classifier.run(docs[1])
        print(classes)
        assert MyClasses.COMPANY_CATEGORY_RPA is classes[0]

    def test_department_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        department_classifier: DepartmentClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = department_classifier.run(docs[1])
        print(classes)
        assert MyClasses.DEPARTMENT_RPA_DEVELOPMENT is classes[0]

    def test_industry_classifier(
        self,
        interview: Path,
        splitter: Splitter,
        industry_classifier: IndustryClassifier,
    ):
        text = interview.read_text()
        docs = splitter.split(interview.stem, text)
        classes = industry_classifier.run(docs[1])
        print(classes)
        assert MyClasses.INDUSTRY_RPA_SOFTWARE is classes[0]

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

        pipe = pipeline.rung()
        doc1p = next(pipe)
        doc2p = next(pipe)
        assert doc1p.metadata["facts"] == doc1.metadata["facts"]
        assert doc2p.metadata["facts"] == doc2.metadata["facts"]
        assert doc2p.metadata["facts"] != doc1p.metadata["facts"]
