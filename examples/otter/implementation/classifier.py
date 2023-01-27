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
