from vector_store import VectorStore

def test_vector_store():
    vs = VectorStore()
    vs.add([0.1]*768, {'id': 1})
    results = vs.search([0.1]*768)
    assert len(results) > 0
