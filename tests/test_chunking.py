# tests/test_chunking.py
from app.utils import sliding_window_chunk, approximate_token_len

def test_chunking_overlap():
    text = " ".join(["word"] * 4000)  # large body
    chunks = sliding_window_chunk(text, chunk_size_tokens=1000, overlap_ratio=0.1, meta={"source":"x"})
    assert len(chunks) >= 3
    # Ensure metadata carried
    assert all("source" in c["metadata"] for c in chunks)
    # Approx token sizes near target
    lens = [approximate_token_len(c["text"]) for c in chunks]
    assert all(600 <= l <= 1400 for l in lens)  # generous bounds
