"""Set geometry: the sources a set spans, and what lies on or off them.

Three reads, one mechanism. A set of embeddings spans a few sources, and each source occupies
a coherent subspace: the directions standing above its own noise floor. Ask how much of a
*target* lies inside the source it fits best and you have grounding. Ask how much of each
*member* lies outside every source once it is removed from its own and you have anomaly. Ask
how much of a *new item* lies outside every source and you have novelty. An injected
instruction and a genuinely new source are the same object read with opposite sign: a mode
off the structure the set already has.

Why one subspace per source
---------------------------
A centroid points one way, and a context assembled from several documents does not. A single
subspace spanning every source at once does better, but it scores an item by how much of it
lies in the span of all of them together, which is a looser question than whether it belongs
to one of them. A nearest-neighbour read asks the local question. Reading each source on its
own asks the local question with no neighbour count to choose. ``research/PAPER.md`` has the
construction and the measured margins over the single subspace, the centroid and k-NN.

How the sources are found
-------------------------
:func:`coherent_basis` gives the set's subspace: the mean direction and the resolved
directions about it, so the directions and their count come from the same frame. Its
dimension is the number of sources. :func:`source_bases` splits the items into that many
groups with no setting: column-pivoted QR on the items' coordinates picks one representative
item per dimension, the polar factor of those rows rotates the coordinates onto them, and
each item joins its largest rotated coordinate (Damle, Minden and Ying, *Simple, direct and
efficient multi-way spectral clustering*, 2019). Each group then gets its own subspace,
built the same way from the group alone.

What this is not
----------------
**Coherence is not factuality.** It reads topic. A wrong-but-on-topic answer reads as
coherent, because staying on a subject and being right about it are different
properties and only the first is visible in the geometry. Use it for off-source and
lost-thread monitoring. Do not pitch it as a factual abstain signal.

**Anomaly is a score and a localization.** It exposes no binary "does this set contain a
foreign item". From one set alone that is ill-posed —
the most peripheral legitimate item of a clean multi-topic set and a genuinely foreign
item of an easier set carry overlapping absolute anomaly. What is exposed is the
per-item score, the ranking, and the top-1 "inspect this first". A calibrated binary
decision needs a reference distribution of known-clean anomalies, which the caller
supplies; :func:`anomaly_z` is that form.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import engine
from .engine import CutUnavailable

__all__ = [
    "coherent_basis",
    "source_bases",
    "in_subspace_fraction",
    "subspace_overlap",
    "chance_overlap",
    "coherence",
    "grounding",
    "anomaly",
    "anomaly_rank",
    "most_anomalous",
    "anomaly_z",
    "novelty_score",
    "assess_coherence",
    "assess_novelty",
]

_UNIT_NORM_FLOOR = 1e-9


def _unit_rows(E) -> np.ndarray:
    """The set's rows as unit directions; raises where a row has no direction."""
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) < 2:
        raise CutUnavailable(f"a {E.shape} set has no subspace to read")
    if not np.isfinite(E).all():
        raise CutUnavailable("the set carries NaN or infinite coordinates")
    n = np.linalg.norm(E, axis=1, keepdims=True)
    if (n == 0).any():
        raise CutUnavailable("a zero-length row has no direction to read")
    return E / n


def _target(v, d: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size != d:
        raise CutUnavailable(f"the target is {v.size}-dimensional and the set {d}")
    if not np.isfinite(v).all() or not np.any(v):
        raise CutUnavailable("the target has no finite direction to read")
    return v


def _mean_and_modes(En: np.ndarray, K: int) -> np.ndarray:
    """The mean direction and the top ``K`` directions about it, orthonormalized."""
    mu = En.mean(0)
    if K == 0:
        return (mu / (np.linalg.norm(mu) + _UNIT_NORM_FLOOR))[:, None]
    _, _, Wt = np.linalg.svd(En - mu, full_matrices=False)
    Q, _ = np.linalg.qr(np.column_stack([mu, Wt[:K].T]))
    return Q


def coherent_basis(E):
    """``(V, K)`` — the coherent subspace of a row-set ``E`` ``(n, d)``.

    ``V`` is ``(d, K)`` with orthonormal columns: the mean direction of the
    direction-normalized set, and the directions about the mean that stand above the noise
    floor. Rows are normalized to unit length first so the subspace is set by where the
    items point, not by which of them happens to have the largest norm.

    The engine reads the count on the whitened frame, which centres each channel on its
    mean, so it counts directions about the mean. The directions are taken from that same
    centred set, and the mean direction the centring removes is added back: for a set of
    embeddings it is the shared topic, the leading coherent mode. ``K`` is therefore the
    engine's count plus one. A set with nothing above the floor about its mean is its mean
    direction alone, ``K = 1``.
    """
    En = _unit_rows(E)
    k = min(engine.resolved_count(En), min(En.shape) - 1)
    V = _mean_and_modes(En, k)
    return V, V.shape[1]


def _group_basis(En: np.ndarray) -> np.ndarray:
    """One source's subspace, built as :func:`coherent_basis` builds the set's. A group with
    nothing above the floor about its mean, or too small to read, is its mean direction
    alone."""
    if len(En) < 2:
        return _mean_and_modes(En, 0)
    try:
        k = min(engine.resolved_count(En), min(En.shape) - 1)
    except CutUnavailable:
        return _mean_and_modes(En, 0)
    return _mean_and_modes(En, k)


def _pivots(A: np.ndarray, k: int) -> np.ndarray:
    """The first ``k`` column pivots of a column-pivoted QR of ``A``: at each step the
    column with the largest norm left after projecting out the columns already chosen."""
    R = np.array(A, dtype=np.float64, copy=True)
    piv = []
    for _ in range(k):
        j = int(np.argmax((R * R).sum(0)))
        piv.append(j)
        q = R[:, j] / (np.linalg.norm(R[:, j]) + _UNIT_NORM_FLOOR)
        R -= np.outer(q, q @ R)
    return np.array(piv)


def _assign(En: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Each item's source: its largest coordinate after rotating onto one pivot item per
    dimension of ``V``."""
    U, _, _ = np.linalg.svd(En @ V, full_matrices=False)
    k = U.shape[1]
    a, _, bt = np.linalg.svd(U[_pivots(U.T, k)].T, full_matrices=False)
    return np.argmax(np.abs(U @ (a @ bt)), axis=1)


def _sources(En: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    V, K = coherent_basis(En)
    g = _assign(En, V)
    return g, np.unique(g), K


def source_bases(E) -> list[np.ndarray]:
    """The subspace of each source the set ``E`` spans, one ``(d, k_s)`` array per source.

    The number of sources is the dimension of :func:`coherent_basis`; the split and each
    source's subspace are built as the module docstring describes.
    """
    En = _unit_rows(E)
    g, labels, _ = _sources(En)
    return [_group_basis(En[g == c]) for c in labels]


def in_subspace_fraction(v, V) -> float:
    """``[0, 1]`` — the fraction of ``v``'s energy lying inside subspace ``V``.

    1 = entirely explained by those directions, 0 = entirely off them.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    vn = v / (np.linalg.norm(v) + _UNIT_NORM_FLOOR)
    proj = np.asarray(V, dtype=np.float64).T @ vn
    return float(np.dot(proj, proj))


def subspace_overlap(V_a, V_b) -> float:
    """``[0, 1]`` — how much of subspace ``V_a`` is covered by subspace ``V_b``.

    ``||V_a^T V_b||_F^2 / k_a``: the mean squared principal cosine between the two, a
    subspace-to-subspace generalization of :func:`in_subspace_fraction`. Use it when
    both sides are sets — one context's sources against another's, or the same context
    before and after a turn.

    Read it against :func:`chance_overlap`, always. In high dimension two subspaces
    that are not specifically aligned are near-orthogonal by default, so a raw overlap
    is uninterpretable and a raw *non*-overlap is uninformative — a read that reports
    "~100% of the structure lies outside" for a random matrix as readily as for a real
    one is measuring the ambient dimension, not the relationship.
    """
    A = np.asarray(V_a, dtype=np.float64)
    B = np.asarray(V_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise CutUnavailable(f"subspaces must share their ambient dim; got {A.shape}, {B.shape}")
    M = A.T @ B
    return float((M * M).sum() / max(1, A.shape[1]))


def chance_overlap(k_b: int, d: int) -> float:
    """``k_b / d`` — what :func:`subspace_overlap` returns for an unrelated ``V_b``.

    The analytic null: a random ``k_b``-dimensional subspace covers ``k_b/d`` of
    anything. Report the ratio of measured overlap to this, not the overlap alone.
    """
    return float(k_b) / float(max(1, d))


def coherence(target_emb, context_embs) -> float:
    """``[0, 1]`` — how much of ``target_emb`` lies in the context source it fits best.

    The largest :func:`in_subspace_fraction` over :func:`source_bases`. Higher is more
    on-topic; a sharp drop flags drift. Topic coherence only — see the module docstring.
    """
    bases = source_bases(context_embs)
    t = _target(target_emb, bases[0].shape[0])
    return max(in_subspace_fraction(t, V) for V in bases)


#: The same read under the name the verification use-case calls it. One implementation.
grounding = coherence


def anomaly(cand_embs) -> np.ndarray:
    """``(n,)`` in ``[0, 1]`` — per item, the energy outside every source of the set, read
    with the item removed from its own.

    High = a foreign mode off the structure the rest of the set has. An item is never read
    against a mode it supplied: its own source is rebuilt without it, and the other sources
    never contained it. An item that forms a source of its own is read against the others
    alone. Reading an item against a subspace that contains it is measured in
    ``research/PAPER.md`` and loses to both the centroid and k-NN.

    Cost: one decomposition per item, for the source it is removed from.
    """
    En = _unit_rows(cand_embs)
    g, labels, _ = _sources(En)
    full = {c: _group_basis(En[g == c]) for c in labels}
    idx = np.arange(len(En))
    out = np.empty(len(En))
    for i in idx:
        best = 0.0
        for c in labels:
            if c == g[i]:
                rest = En[(g == c) & (idx != i)]
                if len(rest) == 0:
                    continue
                V = _group_basis(rest)
            else:
                V = full[c]
            best = max(best, in_subspace_fraction(En[i], V))
        out[i] = 1.0 - best
    return out


def anomaly_rank(cand_embs) -> np.ndarray:
    """Item indices, most anomalous first. ``anomaly_rank(E)[0]`` is the most foreign."""
    return np.argsort(anomaly(cand_embs))[::-1]


def most_anomalous(cand_embs):
    """The index of the single most anomalous item, or ``None`` for an empty set.

    Top-1 localization — which item to inspect first. Not a verdict that the set
    contains a foreign item; see the module docstring for why that question is not well
    posed from one set.
    """
    E = np.asarray(cand_embs, dtype=np.float64)
    if len(E) == 0:
        return None
    return int(np.argmax(anomaly(E)))


def anomaly_z(cand_embs, reference_anomalies) -> np.ndarray:
    """``(n,)`` — per-item anomaly in standard deviations above a reference distribution.

    ``reference_anomalies`` is a sample of :func:`anomaly` values taken over sets known
    to be clean, from the same corpus and embedder. This is the only sound route to a
    binary decision: the caller brings the reference, sets the cut, and owns the
    false-alarm rate that follows.

    Raises rather than inventing a scale when the reference has none — a reference of
    one value, or of identical values, cannot standardize anything, and returning
    zeros or infinities would read as a confident answer.
    """
    ref = np.asarray(reference_anomalies, dtype=np.float64).ravel()
    if ref.size < 2:
        raise CutUnavailable(
            f"a reference of {ref.size} value(s) has no spread to standardize against"
        )
    sd = float(ref.std(ddof=1))
    if sd <= 0.0:
        raise CutUnavailable(
            "the reference anomalies are identical; there is no scale to report z against"
        )
    return (anomaly(cand_embs) - float(ref.mean())) / sd


def novelty_score(new_emb, corpus_embs) -> float:
    """``[0, 1]`` — how much of ``new_emb`` lies off every source of the corpus.

    ``1 - coherence(new_emb, corpus_embs)``. 1 = entirely foreign, and so genuinely new;
    0 = explained by a source already there, and so corroboration rather than information.
    """
    return float(1.0 - coherence(new_emb, corpus_embs))


def _assess(emb, embs, key: str, size_key: str, read) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False, key: None, size_key: 0, "sources": 0, "subspace_dim": 0,
    }
    S = np.asarray(embs, dtype=np.float64)
    info[size_key] = len(S)
    try:
        En = _unit_rows(S)
        g, labels, K = _sources(En)
        t = _target(emb, En.shape[1])
    except CutUnavailable:
        return info
    bases = [_group_basis(En[g == c]) for c in labels]
    score = read(max(in_subspace_fraction(t, V) for V in bases))
    info.update({"available": True, key: round(float(score), 4),
                 "sources": len(bases), "subspace_dim": int(K)})
    return info


def assess_coherence(target_emb, context_embs) -> dict[str, Any]:
    """:func:`coherence` with its evidence, and never raising.

    Returns numbers only — turning a score into a label (grounded / drifted) is a
    display concern and is left to the consumer, because the cut point belongs to
    whoever owns the consequence of being wrong.

    ``available`` False means the read did not happen: the context has no spectrum the
    engine can read, or the target has no direction to compare. The caller must then do
    something other than treat ``grounding`` as 0. ``subspace_dim`` is the dimension of
    :func:`coherent_basis`, and ``sources`` the number of sources the split found in it.
    """
    return _assess(target_emb, context_embs, "grounding", "context_size", lambda f: f)


def assess_novelty(new_emb, corpus_embs) -> dict[str, Any]:
    """:func:`novelty_score` with its evidence, and never raising. See :func:`assess_coherence`."""
    return _assess(new_emb, corpus_embs, "novelty_score", "corpus_size", lambda f: 1.0 - f)
