"""The set-geometry reads: the count, the sources, and the one path each read takes.

The sets are clean orthogonal topics with isotropic noise, so the right answer is known: ``T``
topics are ``T`` sources, and an item drawn off every topic is the most anomalous.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


from entroptics_context.subspace import (anomaly, assess_coherence, assess_novelty,
                                         coherence, coherent_basis, most_anomalous,
                                         novelty_score, source_bases)

D = 64


def topics(weights, seed=1, noise=0.15):
    rng = np.random.default_rng(seed)
    axes = np.linalg.qr(rng.standard_normal((D, len(weights))))[0].T
    E = np.vstack([axes[t] + noise * rng.standard_normal((n, D)) for t, n in enumerate(weights)])
    return E, rng.standard_normal(D)


@pytest.mark.parametrize("weights", [[12], [30], [12, 12, 12], [4, 8, 16], [10, 10, 10, 10, 10]])
def test_t_clean_topics_resolve_t_directions_and_t_sources(weights):
    E, _ = topics(weights)
    _, K = coherent_basis(E)
    assert K == len(weights)
    assert len(source_bases(E)) == len(weights)


def test_the_mean_direction_is_in_the_basis():
    E, _ = topics([12, 12, 12])
    V, _ = coherent_basis(E)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    mu = En.mean(0) / np.linalg.norm(En.mean(0))
    assert np.linalg.norm(V.T @ mu) == pytest.approx(1.0)


@pytest.mark.parametrize("weights", [[12], [12, 12, 12], [4, 8, 16], [10, 10, 10, 10, 10]])
def test_a_planted_item_is_the_most_anomalous(weights):
    E, off = topics(weights)
    assert most_anomalous(np.vstack([E, off])) == len(E)


def test_an_item_is_not_read_against_a_mode_it_supplied():
    """A planted item forms a source of its own. Read against a subspace containing it, it
    would score 0; removed from its own, it is read against the other sources only."""
    E, off = topics([12, 12, 12])
    s = anomaly(np.vstack([E, off]))
    assert s[-1] > 0.5
    assert s[-1] > s[:-1].max()


def test_on_topic_reads_inside_and_off_topic_outside():
    E, off = topics([12, 12, 12])
    assert coherence(E[0], E) > coherence(off, E)
    assert novelty_score(off, E) == pytest.approx(1.0 - coherence(off, E))


def test_assess_reports_the_read_or_that_it_did_not_happen():
    E, off = topics([12, 12, 12])
    a = assess_coherence(off, E)
    assert a["available"] and a["sources"] == 3 and a["subspace_dim"] == 3
    assert a["grounding"] == pytest.approx(round(coherence(off, E), 4))
    n = assess_novelty(off, E)
    assert n["novelty_score"] == pytest.approx(round(1.0 - coherence(off, E), 4))
    assert assess_coherence(off, E[:1])["available"] is False


def test_no_public_read_takes_an_optional_argument():
    """One path per read, across the package's public API: no flag, no default, no knob."""
    import entroptics_context as pkg
    for name in pkg.__all__:
        f = getattr(pkg, name)
        if inspect.ismodule(f) or inspect.isclass(f):
            continue
        if not callable(f):
            continue
        for p in inspect.signature(f).parameters.values():
            assert p.default is inspect.Parameter.empty, f"{name}({p.name}=...) has a default"


def test_pure_noise_is_its_mean_direction_alone():
    """Nothing stands above the floor about the mean, so no direction is added for it."""
    rng = np.random.default_rng(3)
    for n in (10, 40, 200):
        _, K = coherent_basis(rng.standard_normal((n, D)))
        assert K == 1


def test_identical_rows_read_as_their_one_direction():
    rng = np.random.default_rng(4)
    E = np.tile(rng.standard_normal(D), (10, 1))
    t = rng.standard_normal(D)
    cos2 = float((E[0] @ t) ** 2 / (E[0] @ E[0]) / (t @ t))
    assert coherence(t, E) == pytest.approx(cos2)


@pytest.mark.parametrize(
    "case", ["nan_context", "zero_row", "short_target", "nan_target", "zero_target"])
def test_assess_reports_unavailable_rather_than_raising(case):
    E, off = topics([12, 12, 12])
    E = E.copy()
    if case == "nan_context":
        E[0, 0] = np.nan
    elif case == "zero_row":
        E[0] = 0.0
    elif case == "short_target":
        off = off[:5]
    elif case == "nan_target":
        off = np.full(D, np.nan)
    elif case == "zero_target":
        off = np.zeros(D)
    assert assess_coherence(off, E)["available"] is False
    assert assess_novelty(off, E)["available"] is False
