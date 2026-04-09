from medalign.rag.hybrid import reciprocal_rank_fusion


def test_rrf_orders_overlap_first():
    dense = [1, 2, 3, 4]
    sparse = [3, 1, 5, 6]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [d for d, _ in fused]
    # 1 and 3 appear in both lists near the top → should rank first
    assert ids[0] in (1, 3) and ids[1] in (1, 3)
