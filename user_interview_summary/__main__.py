import signal
import sys
from pathlib import Path

from user_interview_summary.classify.classes import Classes
from user_interview_summary.embed.embedder import Embedder
from user_interview_summary.pipeline import Pipeline
from user_interview_summary.query.querier import Querier

signal.signal(signal.SIGINT, lambda _s, _f: sys.exit(1))


def populate():
    try:
        Embedder.create_index()
    except Exception:
        pass

    interviews = (Path(__file__).parent.parent / "interviews").glob("*.txt")
    pipeline = Pipeline(persist=True)
    pipeline.runp(map(Path.open, interviews))


def query():
    querier = Querier()
    res = querier.query(
        "What is the best product for RPA?",
        n=10,
        classes=[],
        with_static=False,
        debug=True,
    )
    print(res)


query()
