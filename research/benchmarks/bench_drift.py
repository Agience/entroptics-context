"""Deciding whether a continuation belongs to a context that spans several sources.

The read
--------
`subspace.coherence` asks how much of a target lies inside the context source it fits best.
On-topic material lies in one of them; a continuation that has wandered lies in none. Nothing
is trained and no threshold is set: the sources and each one's subspace are what the context
itself resolves above its own noise floor.

The baselines are the same question asked with one vector, cosine to the context's centroid,
and asked locally, mean similarity to the k nearest context items.

Why the source count is the axis
--------------------------------
A centroid can only point one way. Given a context assembled from several documents it sits
between them, in a direction none of them occupies, and its distance to a legitimate item
then grows with the spread of the context rather than with anything about the item. One
subspace per source keeps on-topic material inside its own source however many topics the
context covers.

A retrieval-augmented context built from several documents, or a conversation that has
covered several subjects, is the ordinary case. Measuring only at one source measures the
regime where a centroid is by construction sufficient.

Run::

    python -m research.benchmarks.bench_drift
    python -m research.benchmarks.bench_drift --topics 1,2,3,5,8 --trials 300
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from entroptics_context import coherence, coherent_basis, in_subspace_fraction, source_bases
from entroptics_context.engine import CutUnavailable
from research.benchmarks.bench_injection import articles, auroc, boot

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--topics", default="1,2,3,5")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--per-source", type=int, default=12,
                    help="sentences taken from each source to build the context")
    ap.add_argument("--held", type=int, default=6,
                    help="held-out on-topic sentences per source. AUROC over t targets moves "
                         "in steps of 1/(t/2)^2, so too few quantizes the read: at 2 per "
                         "source and M=1 it can only take the values 0, .25, .5, .75, 1.")
    ap.add_argument("--min-sentences", type=int, default=20)
    ap.add_argument("--knn", type=int, default=5,
                    help="neighbours for the k-NN baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    Ms = [int(x) for x in args.topics.split(",")]

    from sentence_transformers import SentenceTransformer

    arts = articles(args.min_sentences, max(80, max(Ms) * 30))
    model = SentenceTransformer(args.encoder, device="cpu")
    model.eval()
    print(f"encoder {args.encoder} | {len(arts)} articles | {args.trials} trials per topic count",
          flush=True)
    embs = [np.asarray(model.encode(s, normalize_embeddings=True, show_progress_bar=False),
                       dtype=np.float64) for s in arts]

    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 74)
    print("IS THIS CONTINUATION ON-TOPIC FOR A CONTEXT SPANNING M SOURCES?")
    print("=" * 74)
    print(f"\n{'sources':>8} {'modes':>6} {'ctx':>5} | {'entroptics':>10} {'centroid':>9} {'k-NN':>7} "
          f"{'vs centroid [95% CI]':>23} {'vs k-NN [95% CI]':>23}")
    print("-" * 74)
    rows = []
    for M in Ms:
        sub, cen, knn, Ks, srcs, sizes = [], [], [], [], [], []
        one = []
        for _ in range(args.trials):
            idx = rng.choice(len(arts), size=M + 1, replace=False)
            ctx_src, off_src = idx[:M], idx[M]
            ctx, held = [], []
            for a in ctx_src:
                e = embs[a]
                take = min(args.per_source, len(e) - args.held)
                ctx.append(e[:take])
                held.append(e[take:take + args.held])  # on-topic, and NOT in the context
            C = np.vstack(ctx)
            on = np.vstack(held)
            off = embs[off_src][:len(on)]              # a different article: genuinely off-topic
            if len(off) == 0:
                continue
            try:
                _, K = coherent_basis(C)
                n_src = len(source_bases(C))
            except CutUnavailable:
                continue
            centroid = C.mean(0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            T = np.vstack([on, off])
            label = np.r_[np.ones(len(on), bool), np.zeros(len(off), bool)]
            s_sub = np.array([coherence(t, C) for t in T])
            s_cen = T @ centroid
            # k-NN: the same target scored by its similarity to the k nearest CONTEXT items.
            # A local baseline, so it carries no penalty for a context that spans topics.
            k = min(args.knn, len(C))
            s_knn = np.sort(T @ C.T, axis=1)[:, -k:].mean(1)
            sub.append(auroc(s_sub, label))
            cen.append(auroc(s_cen, label))
            knn.append(auroc(s_knn, label))
            # One span over every source, the read of PAPER section 2's first half. Section 6
            # reads this arm; it is not a library path.
            V, _ = coherent_basis(C)
            one.append(auroc(np.array([in_subspace_fraction(t, V) for t in T]), label))
            Ks.append(K)
            srcs.append(n_src)
            sizes.append(len(C))
        d = np.array(sub) - np.array(cen)
        mm, lo, hi = boot(d)
        km, klo, khi = boot(np.array(sub) - np.array(knn))
        rows.append({"M": M, "modes": float(np.mean(Ks)), "sources": float(np.mean(srcs)),
                     "n": len(sub),
                     "subspace": float(np.mean(sub)), "centroid": float(np.mean(cen)),
                     "knn": float(np.mean(knn)),
                     "edge": mm, "ci": [lo, hi],
                     "edge_vs_knn": km, "ci_vs_knn": [klo, khi],
                     "one_span": float(np.mean(one)),
                     "one_span_vs_knn": list(boot(np.array(one) - np.array(knn)))})
        print(f"{M:>8} {np.mean(Ks):>6.1f} {np.mean(sizes):>5.0f} | "
              f"{np.mean(sub):>10.3f} {np.mean(cen):>9.3f} {np.mean(knn):>7.3f} "
              f"{mm:>+8.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{km:>+8.3f} [{klo:+.3f},{khi:+.3f}]", flush=True)
    print("-" * 74)
    if len(rows) > 1:
        print(f"edge at {rows[0]['M']} source(s): {rows[0]['edge']:+.3f}   "
              f"at {rows[-1]['M']}: {rows[-1]['edge']:+.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"encoder": args.encoder, "trials": args.trials, "rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
