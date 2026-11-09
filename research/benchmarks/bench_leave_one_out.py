"""Top-1 localization of an injected item under both ``anomaly`` reads, on clean topic sets.

The construction. ``T`` orthogonal topic directions in ``d = 192``, eight items around
each, noise of norm 0.25 per item. One injected item is appended, a unit vector whose
foreignness ``f`` is its amplitude on a direction no topic uses:
``sqrt(1 - f^2) * topic_0 + f * fresh``. ``f = 0.7`` keeps most of it on a topic,
``f = 1.0`` is fully orthogonal. Each cell is the fraction of draws in which
``most_anomalous`` names the injected item, under the in-set read
(``leave_one_out=False``) and the default.

The in-set read builds the span from every item, the injected one included; the
default leaves the scored item out first. The count is read on the whitened frame,
which centres each channel on its mean, so from three topics up ``T`` orthogonal topics
resolve ``T - 1`` modes. The second table prints that count and, under each read, the
injected item's anomaly next to the highest anomaly among the other items. The results
are not monotone in the topic count.

Nothing here is fitted. ``f``, ``T``, the noise norm and the item count describe the
construction; the reads carry no constant.

Run::

    python -m research.benchmarks.bench_leave_one_out
    python -m research.benchmarks.bench_leave_one_out --draws 50
"""
from __future__ import annotations

import argparse

import numpy as np

from entroptics_context import anomaly, coherent_basis, most_anomalous


def draw(rng, topics: int, foreignness: float, *, d: int, per: int, spread: float):
    B = np.linalg.qr(rng.standard_normal((d, topics + 1)))[0]
    scale = spread / np.sqrt(d)
    E = np.concatenate([B[:, t] + scale * rng.standard_normal((per, d)) for t in range(topics)])
    inj = np.sqrt(1.0 - foreignness ** 2) * B[:, 0] + foreignness * B[:, topics]
    return np.vstack([E, inj]), len(E)


def hit_rates(topics, foreignness, *, draws, seed, d, per, spread):
    loo = ins = 0
    for s in range(draws):
        E, target = draw(np.random.default_rng(seed + s), topics, foreignness,
                         d=d, per=per, spread=spread)
        loo += int(most_anomalous(E) == target)
        ins += int(most_anomalous(E, leave_one_out=False) == target)
    return ins / draws, loo / draws


def evidence(topics, foreignness, *, draws, seed, d, per, spread):
    """Per cell, means over draws: the count on the set without the injected item, and under
    each read the injected item's score next to the highest score among the other items."""
    k, rows = [], {"in-set": ([], []), "leave-one-out": ([], [])}
    for s in range(draws):
        E, target = draw(np.random.default_rng(seed + s), topics, foreignness,
                         d=d, per=per, spread=spread)
        k.append(coherent_basis(E[:target])[1])
        for name, loo in (("in-set", False), ("leave-one-out", True)):
            a = anomaly(E, leave_one_out=loo)
            rows[name][0].append(a[target])
            rows[name][1].append(a[:target].max())
    return float(np.mean(k)), {n: (float(np.mean(x)), float(np.mean(y))) for n, (x, y) in rows.items()}


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

    print(f"top-1 localization of one injected item, {args.draws} draws, d={args.dim}, "
          f"{args.per} items per topic\n")
    print(f"{'foreignness':>12} {'read':>14}" + "".join(f"{f'{t} topic' + ('s' if t > 1 else ''):>10}"
                                                       for t in topics))
    for f in fs:
        cells = [hit_rates(t, f, draws=args.draws, seed=args.seed, d=args.dim,
                           per=args.per, spread=args.spread) for t in topics]
        print(f"{f:>12.1f} {'in-set':>14}" + "".join(f"{c[0]:>10.2f}" for c in cells))
        print(f"{'':>12} {'leave-one-out':>14}" + "".join(f"{c[1]:>10.2f}" for c in cells))

    print()
    print("what each read scores, means over the same draws: K is the count on the set "
          "without the injected item; 'item' is the injected item's anomaly, 'other' the "
          "highest anomaly among the rest")
    print()
    print(f"{'foreignness':>12} {'topics':>7} {'K':>5} {'read':>14} {'item':>7} {'other':>7}")
    for f in fs:
        for t in topics:
            k, ev = evidence(t, f, draws=args.draws, seed=args.seed, d=args.dim,
                             per=args.per, spread=args.spread)
            for name, (it, ot) in ev.items():
                print(f"{f:>12.1f} {t:>7d} {k:>5.1f} {name:>14} {it:>7.2f} {ot:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
