from chunking import chunk_text

def test_chunking():
    text = 'a b c d e f g h'
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    assert len(chunks) > 0
