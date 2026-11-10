"""Reading a context that spans several sources as a subspace rather than a centre.

A context assembled from several documents — a retrieval-augmented prompt, an agent
scratchpad, a long conversation — has no single direction. Its centroid sits between its
sources, in a direction none of them occupies, and every guard built on that centroid gets
worse as the context spreads. Each source the context resolves above its own noise floor
keeps a subspace of its own instead, so legitimate material lies inside its source however
many topics the context covers and foreign material lies in none. Nothing is trained, no
threshold is set, no constant is fitted per corpus, and no function takes an optional
argument.

Quick start
-----------
::

    from entroptics_context import source_bases, most_anomalous, coherence

    sources = source_bases(context_embeddings)   # one subspace per source
    suspect = most_anomalous(context_embeddings) # the item furthest outside every source
    score = coherence(answer_embedding, context_embeddings)

Where it beats a centroid
-------------------------
The margin grows with the number of sources the context spans. On real prompt injections
the edge over a centroid is +0.070, +0.134 and +0.223 AUROC at 2, 3 and 5 sources, and over
a k-nearest-neighbour detector +0.013, +0.019 and +0.024; on the on-topic decision the edge
over a centroid is +0.056, +0.110 and +0.172, and k-NN ties. The
largest margin is on selection: where relevance spreads across several facets the derived cut
holds F1 at 1.000 through four facets and at 0.915 and 0.711 at five and six, where a cosine
cut falls to 0.000. research/PAPER.md has the construction
and what every number was measured on.

``coherence`` reads topic coherence rather than factual correctness, and the anomaly reads
rank items within one set rather than returning a verdict about it.

Layout
------
``engine``     the one seam onto `entroptics`. Nothing else imports it.
``subspace``   coherent basis, the sources a set spans, anomaly, coherence.
``lock``       ``gap_split`` / ``top_break`` — where a salience spectrum stops.
``cut``        the retrieval budget: a query-relative screen, and the lock on cosine.
               A separate and weaker result; see the scope note in README.md.
"""
from __future__ import annotations

from . import cut, engine, lock, subspace
from .cut import (cosine_salience, derive_heads, head_screen, select,
                  select_by_cosine, select_with_info, signal_power)
from .engine import CutUnavailable, focus, occupancy, resolved_rank
from .lock import gap_split, top_break
from .subspace import (
    anomaly,
    anomaly_rank,
    anomaly_z,
    chance_overlap,
    coherence,
    coherent_basis,
    grounding,
    in_subspace_fraction,
    most_anomalous,
    novelty_score,
    source_bases,
    subspace_overlap,
)

__version__ = "0.3.0"

__all__ = [
    # modules
    "engine", "lock", "cut", "subspace",
    # engine
    "CutUnavailable", "resolved_rank", "occupancy", "focus",
    # the lock
    "gap_split", "top_break",
    # the retrieval cut
    "select", "select_by_cosine", "cosine_salience",
    "select_with_info", "head_screen", "signal_power", "derive_heads",
    # set geometry
    "coherent_basis", "source_bases", "in_subspace_fraction", "subspace_overlap", "chance_overlap",
    "coherence", "grounding", "anomaly", "anomaly_rank", "most_anomalous", "anomaly_z",
    "novelty_score",
]
