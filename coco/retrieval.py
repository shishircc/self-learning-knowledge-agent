import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import cosine_similarity


def authoritativeness_bias(authoritativeness: float, scale: float) -> float:
    """Small linear term added to the RRF final score so high-trust packets
    break ties between equally-relevant candidates. See DESIGN.md §5.10.

    The scale is intentionally tiny (default 0.001) — sharp semantic matches
    should still win over a weakly-relevant but high-trust packet.
    """
    a = max(0.0, float(authoritativeness or 0.0))
    return float(scale) * a


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def _max_cosine_over_facets(query_vec: np.ndarray, packet) -> float:
    if not packet.topics:
        return 0.0
    return max(cosine_similarity(query_vec, fv) for fv in packet.topic_vectors_np())


def _ranks_with_floor(scores, floor: float) -> dict[int, int]:
    """Sort packets by score descending; only those with score > floor get a rank.
    Returns {packet_index: rank} for packets that pass the floor; absent index
    means the packet contributes 0 from this channel.
    """
    sorted_idx = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    out: dict[int, int] = {}
    rank = 0
    for i in sorted_idx:
        if float(scores[i]) > floor:
            out[i] = rank
            rank += 1
    return out


def rrf_packet_search(
    query_text: str,
    query_vec: np.ndarray,
    packets: list,
    k: int = 60,
    top_n: int = 2,
    debug: bool = False,
    cosine_floor: float = 0.0,
):
    """Three-channel RRF over packets:
      A: BM25 over packet's combined topic-facets text
      B: max cosine across packet's topic vectors
      C: BM25 over packet's entity bag

    **Zero-score filtering per channel.** A packet is given a rank in a
    channel only if its raw score passes a per-channel floor:
      - BM25 channels (A, C): score > 0  (no token overlap means no rank).
      - Cosine channel (B):  score > cosine_floor  (default 0.0; tune up
        to e.g. 0.1 to discard near-orthogonal "noise" matches).
    Packets without a rank in a channel contribute exactly 0 from that
    channel — so truly irrelevant packets can drop to 0 if they miss
    every channel, instead of getting near-rank-1 RRF noise.

    When debug=True, prints a per-packet breakdown showing each channel's
    raw score, rank (or "filtered"), and RRF contribution.
    """
    if not packets:
        return []

    n = len(packets)
    q_tokens = tokenize(query_text)

    # Channel A: topic BM25
    topic_corpus = [tokenize(p.combined_topic_text()) for p in packets]
    bm25_topic = BM25Okapi(topic_corpus)
    a_scores = bm25_topic.get_scores(q_tokens)
    a_rank_of = _ranks_with_floor(a_scores, 0.0)

    # Channel B: max-cosine across facets (floor applied)
    b_scores = np.array([_max_cosine_over_facets(query_vec, p) for p in packets])
    b_rank_of = _ranks_with_floor(b_scores, cosine_floor)

    # Channel C: entity BM25 (packet's "document" is its lowercased entity bag)
    entity_corpus = [list(p.entities) if p.entities else [""] for p in packets]
    bm25_entity = BM25Okapi(entity_corpus)
    c_scores = bm25_entity.get_scores(q_tokens)
    c_rank_of = _ranks_with_floor(c_scores, 0.0)

    rrf = {i: 0.0 for i in range(n)}
    for i, rank in a_rank_of.items():
        rrf[i] += 1.0 / (k + rank + 1)
    for i, rank in b_rank_of.items():
        rrf[i] += 1.0 / (k + rank + 1)
    for i, rank in c_rank_of.items():
        rrf[i] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_n]

    if debug:
        print(
            f"  retrieval (k={k}, cosine_floor={cosine_floor}, "
            f"{n} candidate(s)) — query tokens: {q_tokens}",
            flush=True,
        )

        def _line(raw, rank_of, ranked_count, idx):
            if idx in rank_of:
                rank = rank_of[idx]
                contrib = 1.0 / (k + rank + 1)
                return f"raw={raw:.3f}  rank={rank + 1}/{ranked_count}  contrib=+{contrib:.4f}"
            return f"raw={raw:.3f}  filtered (below floor) contrib=+0.0000"

        a_count = len(a_rank_of)
        b_count = len(b_rank_of)
        c_count = len(c_rank_of)
        for i, score in ranked:
            pkt = packets[i]
            topic = pkt.topics[0].text if pkt.topics else pkt.id
            print(
                f"    [final {score:.4f}] {pkt.id} \"{topic}\"\n"
                f"        A topic-BM25:  {_line(float(a_scores[i]), a_rank_of, a_count, i)}\n"
                f"        B max-cosine:  {_line(float(b_scores[i]), b_rank_of, b_count, i)}\n"
                f"        C entity-BM25: {_line(float(c_scores[i]), c_rank_of, c_count, i)}",
                flush=True,
            )

    return [(packets[i], score) for i, score in ranked]


def resolve_topic(new_topic_vec: np.ndarray, session_topics: list[dict], threshold: float):
    """Match the new turn's topic facet against session topic list.
    Returns (is_new_topic, matched_idx_or_None).
    Pure cosine since session topics list is small.
    """
    if not session_topics:
        return True, None
    best_idx = -1
    best_score = -1.0
    for i, t in enumerate(session_topics):
        v = np.asarray(t["topic_vector"], dtype=np.float32)
        s = cosine_similarity(new_topic_vec, v)
        if s > best_score:
            best_score = s
            best_idx = i
    if best_score >= threshold:
        return False, best_idx
    return True, None


def best_packet_facet_match(query_vec: np.ndarray, packets: list):
    """Find the packet whose best facet most closely matches the query vector.
    Returns (best_packet, best_score). Used for new-facet-vs-new-packet decision.
    """
    if not packets:
        return None, -1.0
    best_packet = None
    best_score = -1.0
    for p in packets:
        s = _max_cosine_over_facets(query_vec, p)
        if s > best_score:
            best_score = s
            best_packet = p
    return best_packet, best_score


def rank_packet_facet_candidates(
    query_vec: np.ndarray, packets: list, top_n: int = 5
) -> list[tuple]:
    """Rank packets by max-facet cosine; return top_n as [(packet, score, best_facet_text), ...]."""
    if not packets:
        return []
    rows: list[tuple] = []
    for p in packets:
        if not p.topics:
            rows.append((p, 0.0, ""))
            continue
        best_facet_text = p.topics[0].text
        best = -1.0
        for f in p.topics:
            s = cosine_similarity(query_vec, f.vec_np())
            if s > best:
                best = s
                best_facet_text = f.text
        rows.append((p, float(best), best_facet_text))
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]


def best_scratchpad_match(query_vec: np.ndarray, entries: list, threshold: float):
    """Returns (best_entry, score) if best meets threshold, else (None, best_score)."""
    if not entries:
        return None, -1.0
    best = None
    best_score = -1.0
    for e in entries:
        s = cosine_similarity(query_vec, e.topic_vec_np())
        if s > best_score:
            best_score = s
            best = e
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def rank_scratchpad_candidates(
    query_vec: np.ndarray, entries: list, top_n: int = 3
) -> list[tuple]:
    """Rank scratchpad entries by topic cosine; returns [(entry, score), ...]."""
    if not entries:
        return []
    rows = [(e, float(cosine_similarity(query_vec, e.topic_vec_np()))) for e in entries]
    rows.sort(key=lambda r: -r[1])
    return rows[:top_n]
