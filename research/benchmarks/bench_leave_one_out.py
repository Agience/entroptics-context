"""Top-1 localization of an injected item by ``anomaly``, on clean topic sets.

The construction. ``T`` orthogonal topic directions in ``d = 192``, eight items around
each, noise of norm 0.25 per item. One injected item is appended, a unit vector whose
foreignness ``f`` is its amplitude on a direction no topic uses:
``sqrt(1 - f^2) * topic_0 + f * fresh``. ``f = 0.7`` keeps most of it on a topic,
``f = 1.0`` is fully orthogonal. Each cell is the fraction of draws in which
``most_anomalous`` names the injected item.

``anomaly`` reads each item against every source of the set with the item removed from its
own, so the item is never scored inside a span it helped build. The second table prints, per
cell, the count and the number of sources on the set without the injected item, and the
injected item's anomaly next to the highest anomaly among the other items.

Nothing here is fitted. ``f``, ``T``, the noise norm and the item count describe the
construction; the read carries no constant.

Run::

    python -m research.benchmarks.bench_leave_one_out
    python -m research.benchmarks.bench_leave_one_out --draws 50
"""
from __future__ import annotations

import argparse

import numpy as np

from entroptics_context import anomaly, coherent_basis, source_bases


def draw(rng, topics: int, foreignness: float, *, d: int, per: int, spread: float):
    B = np.linalg.qr(rng.standard_normal((d, topics + 1)))[0]
    scale = spread / np.sqrt(d)
    E = np.concatenate([B[:, t] + scale * rng.standard_normal((per, d)) for t in range(topics)])
    inj = np.sqrt(1.0 - foreignness ** 2) * B[:, 0] + foreignness * B[:, topics]
    return np.vstack([E, inj]), len(E)


def cell(topics, foreignness, *, draws, seed, d, per, spread):
    """Means over draws: the hit rate, the count and source count on the set without the
    injected item, the injected item's anomaly, and the highest anomaly among the rest."""
    hit, k, src, item, other = [], [], [], [], []
    for s in range(draws):
        E, target = draw(np.random.default_rng(seed + s), topics, foreignness,
                         d=d, per=per, spread=spread)
        a = anomaly(E)
        hit.append(int(np.argmax(a) == target))
        k.append(coherent_basis(E[:target])[1])
        src.append(len(source_bases(E[:target])))
        item.append(a[target])
        other.append(a[:target].max())
    return tuple(float(np.mean(x)) for x in (hit, k, src, item, other))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=15)
    ap.add_argument("--topics", default="1,3,5")
    ap.add_argument("--foreignness", default="0.7,0.9,1.0")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--per", type=int, default=8, help="items per topic")
    ap.add_argument("--spread", type=float, default=0.25, help="noise norm per item")
    ap.add_argument("--seed", type=int, default=400)
    args = ap.parse_args(argv)
    topics = [int(x) for x in args.topics.split(",")]
    fs = [float(x) for x in args.foreignness.split(",")]
    kw = dict(draws=args.draws, seed=args.seed, d=args.dim, per=args.per, spread=args.spread)
    cells = {(f, t): cell(t, f, **kw) for f in fs for t in topics}

    print(f"top-1 localization of one injected item, {args.draws} draws, d={args.dim}, "
          f"{args.per} items per topic\n")
    print(f"{'foreignness':>12}" + "".join(f"{f'{t} topic' + ('s' if t > 1 else ''):>10}"
                                           for t in topics))
    for f in fs:
        print(f"{f:>12.1f}" + "".join(f"{cells[f, t][0]:>10.2f}" for t in topics))

    print()
    print("means over the same draws: K and sources on the set without the injected item; "
          "'item' is the injected item's anomaly, 'other' the highest anomaly among the rest")
    print()
    print(f"{'foreignness':>12} {'topics':>7} {'K':>5} {'sources':>8} {'item':>7} {'other':>7}")
    for f in fs:
        for t in topics:
            _, k, src, it, ot = cells[f, t]
            print(f"{f:>12.1f} {t:>7d} {k:>5.1f} {src:>8.1f} {it:>7.2f} {ot:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
