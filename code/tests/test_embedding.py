from embedding import generate_embedding

def test_embedding():
    emb = generate_embedding('test')
    assert len(emb) == 768
