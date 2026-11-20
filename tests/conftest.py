import pytest

from payrag.data_gen import generate_corpus
from payrag.vector_store import LocalVectorStore


@pytest.fixture(scope="session")
def corpus():
    return generate_corpus()


@pytest.fixture(scope="session")
def store(corpus):
    return LocalVectorStore(corpus)
