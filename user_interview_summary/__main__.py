from pathlib import Path

from tqdm import tqdm

from user_interview_summary.classify.classes import Classes
from user_interview_summary.embed.embedder import Embedder
from user_interview_summary.pipeline import Pipeline
from user_interview_summary.query.querier import Querier


def populate():
    try:
        Embedder.create_index()
    except:
        pass

    interviews = (Path(__file__).parent.parent / "interviews").glob("*.txt")
    pipeline = Pipeline(persist=True)
    pipe = tqdm(pipeline.run([f.open(mode="r") for f in interviews]))

    for doc in pipe:
        pipe.set_description(f"Processing {doc.metadata['file']}")


def query():
    querier = Querier()
    res = querier.query("What are the hardest processes?", n=10, classes=[])
    print(res)


populate()
