from medalign.data.format import to_chatml, dedup_minhash


def test_chatml_roundtrip():
    s = to_chatml("What is hypertension?", "High blood pressure.")
    assert "<|im_start|>user" in s and "<|im_start|>assistant" in s
    assert "hypertension" in s and "blood pressure" in s


def test_dedup_minhash_removes_duplicates():
    texts = [
        "patient has chest pain and shortness of breath",
        "patient has chest pain and shortness of breath",  # exact dup
        "the capital of france is paris",
    ]
    keep = dedup_minhash(texts, threshold=0.8)
    assert len(keep) == 2
