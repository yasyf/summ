from pinecone import PodSpec

from summ.embed.embedder import Embedder


class FakeIndex:
    def __init__(self):
        self.vectors = None

    def upsert(self, *, vectors):
        self.vectors = vectors


class FakePinecone:
    def __init__(self):
        self.created = None
        self.index_names = []
        self.index_exists = True
        self.target = FakeIndex()

    def create_index(self, **kwargs):
        self.created = kwargs

    def has_index(self, name):
        self.index_names.append(name)
        return self.index_exists

    def index(self, *, name):
        self.index_names.append(name)
        return self.target


def test_embedder_uses_current_client_for_index_lifecycle(monkeypatch):
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    monkeypatch.setattr("summ.embed.embedder.OpenAIEmbeddings", lambda: object())
    client = FakePinecone()
    embedder = Embedder("interviews", pinecone_client=client)

    assert client.index_names == []
    assert embedder.has_index()
    assert embedder.index is client.target

    embedder.create_index()
    assert client.created["name"] == "interviews"
    assert client.created["dimension"] == Embedder.GPT3_DIMS
    assert client.created["spec"] == PodSpec(
        environment="us-east-1-aws",
        metadata_config={"indexed": ["classes"]},
    )
