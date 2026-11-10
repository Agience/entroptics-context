# entroptics-context

[![CI](https://github.com/Agience/entroptics-context/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/entroptics-context/actions/workflows/ci.yml)
[![Licence: AGPL-3.0-only or commercial](https://img.shields.io/badge/licence-AGPL--3.0--only%20or%20commercial-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/entroptics-context)](https://pypi.org/project/entroptics-context/)
[![Python](https://img.shields.io/pypi/pyversions/entroptics-context)](https://pypi.org/project/entroptics-context/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22313961.svg)](https://doi.org/10.5281/zenodo.22313961)
[![Sponsor](https://img.shields.io/badge/sponsor-Agience-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/Agience)

Paper: [research/PAPER.pdf](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.pdf) (two-column; LaTeX source in [research/PAPER.tex](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.tex), Markdown in [research/PAPER.md](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.md)).

Read the context as the sources it resolves, and the selection follows from the data.

A language model's context is assembled from several sources at once: a retrieval-augmented
prompt, an agent scratchpad, a long conversation. The decisions made about it — how many
retrieved items to keep, which of them to trust, whether an answer stayed on its evidence — are
conventionally made with one number per item against one summary of the whole.

This reads the context as the sources it spans, each one the directions it resolves above its
own noise floor, using the [`entroptics`](https://github.com/Agience/entroptics) instrument,
and makes each of those decisions from those spans. The sources and their directions are
counted from the data: nothing is trained, no threshold is set, no constant is fitted per
corpus, and no function takes an optional argument.

---

## The result

**How many to keep, when relevance has several facets.** A query whose relevance spreads across
R independent facets, both sides running the same parameter-free lock over the same screen:

| facets R | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| cosine F1 | 1.000 | 1.000 | 1.000 | 0.806 | 0.000 | 0.000 |
| **entroptics F1** | 1.000 | 1.000 | 1.000 | **1.000** | **0.915** | **0.711** |
| edge | +0.000 | +0.000 | +0.000 | **+0.194** | **+0.915** | **+0.711** |

Cosine falls to 0.000 by five facets; entroptics holds 1.000 through four and keeps 0.915 and
0.711 at five and six. One scalar carries one facet's worth of evidence; an R-mode subspace holds
several.

**Which item is foreign.** Real attacks from `deepset/prompt-injections` planted in wikitext-103
articles, against a centroid and against a k-nearest-neighbour detector:

| sources | **entroptics** | centroid | k-NN | vs centroid | vs k-NN |
|---|---|---|---|---|---|
| 1 | **0.988** | 0.983 | 0.984 | +0.005 [+0.000, +0.010] | **+0.005** [+0.001, +0.008] |
| 2 | **0.972** | 0.902 | 0.959 | +0.070 [+0.055, +0.087] | **+0.013** [+0.008, +0.019] |
| 3 | **0.967** | 0.833 | 0.947 | +0.134 [+0.111, +0.157] | **+0.019** [+0.013, +0.026] |
| 5 | **0.941** | 0.718 | 0.917 | +0.223 [+0.192, +0.254] | **+0.024** [+0.016, +0.032] |

The margin over a centroid grows from +0.005 to +0.223 as sources multiply, and the margin over
k-NN from +0.005 to +0.024, every k-NN interval excluding zero. On the question an operator
asks, which item to open first, entroptics has the highest rate at every source count:
0.785 / 0.450 / 0.410 / 0.175 against the centroid's 0.735 / 0.250 / 0.145 / 0.030 and k-NN's
0.720 / 0.365 / 0.300 / 0.120, almost six times the centroid's hit rate at five sources from a
pool of 191 items.

**Whether a continuation belongs.** Held-out on-topic sentences against sentences from a
different article:

| sources | **entroptics** | centroid | k-NN | vs centroid | vs k-NN |
|---|---|---|---|---|---|
| 1 | 0.959 | 0.960 | 0.966 | −0.002 [−0.007, +0.004] | −0.007 [−0.012, −0.003] |
| 2 | 0.946 | 0.890 | 0.948 | **+0.056** [+0.046, +0.067] | −0.002 [−0.007, +0.003] |
| 3 | 0.937 | 0.827 | 0.936 | **+0.110** [+0.096, +0.123] | +0.001 [−0.003, +0.004] |
| 5 | 0.910 | 0.738 | 0.909 | **+0.172** [+0.158, +0.187] | +0.001 [−0.003, +0.005] |

The same growth from an independent task: the centroid falls 0.960 → 0.738 as the context
spreads while entroptics holds 0.959 → 0.910. Against k-NN it ties from two sources up and
trails by 0.007 at one.

```bash
python -m research.benchmarks.bench_facets       # runs on CPU in seconds, no corpus
python -m research.benchmarks.bench_injection
python -m research.benchmarks.bench_drift
```

[`research/PAPER.pdf`](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.pdf) has the
construction, the real-corpus results on BEIR and HotpotQA, and every number with what it was measured on
(the same text in Markdown is [`research/PAPER.md`](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.md)).

## Use it

```python
from entroptics_context import source_bases, most_anomalous, coherence

sources = source_bases(context_embeddings)     # one subspace per source the context spans
suspect = most_anomalous(context_embeddings)   # the item furthest outside every source
score   = coherence(answer_embedding, context_embeddings)   # is this still on its sources?
```

The number of sources and each one's directions are read from the data, from the singular
values standing above the derived noise floor, so nothing is chosen and no function takes an
optional argument. `context_embeddings` are the ones your pipeline already computed
for retrieval; no additional model call is made.

**These are scores and rankings, not verdicts.** `most_anomalous` localizes the item worth
inspecting. "Is there an anomaly in this one set" is ill-posed without a reference
distribution, so a caller wanting a yes/no supplies its own null.

## Install

```bash
pip install entroptics-context
```

[Python 3.11](https://www.python.org/downloads/) or later. Runtime dependencies are numpy and
[`entroptics` 0.2.12](https://pypi.org/project/entroptics/) or later, nothing else. No encoder, no torch, no corpus: the vectors are yours, and the package never
downloads anything.

Working on it, or reproducing the tables:

```bash
pip install -e ".[dev]"      # + pytest
pytest -q                    # the whole suite, no model, no downloads
pip install -e ".[bench]"    # + encoders and datasets, for research/benchmarks/
pip install -e ".[bench-llm]" # + a language model, for bench_hotpot_e2e
```

## Layout

```
research/
  PAPER.pdf       the paper, typeset.
  PAPER.md        the same write-up in markdown, and every number with what it was measured on.
  benchmarks/
    bench_injection   locating a real injection, against source count.
    bench_drift       on-topic decision, against source count.
    bench_facets      the budget's win condition: relevance across R facets.
    bench_beir        the budget on real BEIR corpora, where relevance is unimodal.
    bench_hotpot_e2e  end to end with a language model answering.
src/entroptics_context/
  engine.py       the one seam onto `entroptics`. Nothing else imports it.
  subspace.py     coherent basis, the sources a set spans, anomaly, coherence.
  lock.py         gap_split / top_break — where a salience spectrum stops.
  cut.py          the retrieval budget: a query-relative screen, and the lock on cosine.
tests/            the suite, including the structural guards that read prose
```

## Where it ties, and why

**On unimodal relevance it ties, which is the same prediction.** At one coherent mode the read
reduces to the statistic it is measured against. Four real BEIR corpora (nfcorpus, scifact,
arguana, fiqa) over 2,421 scored queries all tie with cosine and the same lock: −0.005, −0.013,
−0.001 and −0.003, every interval containing zero. At the point the cut sees them, only 198 of
those queries (8.2%) carry four or more relevant documents — 144 of nfcorpus's 260, 4 of
scifact's 270, none of arguana's 1,357 and 50 of fiqa's 534 — so on three of the four corpora the
deciding distribution has one mode, and a tie is what the law requires there. And HotpotQA
end-to-end with a language model, handed cosine's top-m where m is the count the cut derived for
that question: +0.000 [+0.000, +0.000] on 150 questions with Qwen2.5-1.5B — the selection is
indistinguishable. The BEIR runs use the all-MiniLM-L6-v2 encoder; injection, drift and HotpotQA
use bge-small-en-v1.5.

**Size the budget to the task.** No single fixed count suits every corpus: the best constant k
over the four BEIR corpora (top-1) gives up as much as 0.130 F1 against a corpus's own best fixed
k (nfcorpus, where that is top-10), while the derived count gives up at most 0.020. On HotpotQA
the derived count spends 38% of a fixed top-5's context tokens, keeping 2.1 paragraphs where a
multi-hop answer needs two gold paragraphs, at −0.058 F1 [−0.123, +0.006] against that fixed
top-5. Where a task needs several passages, put a floor on
the count.

**Coherence reads topic, not factual correctness.** A wrong-but-on-topic answer reads as
coherent. On the 88 of those questions that kept more than one paragraph, right answers scored
0.432 and wrong ones 0.420.
Use it for off-source and lost-thread monitoring, not as a fact checker.

## Built on entroptics

Every spectral read reaches [`entroptics`](https://github.com/Agience/entroptics) through one module,
[`engine.py`](src/entroptics_context/engine.py); nothing else imports the library and a test
enforces it. The resolved-mode count that sets `K` is the library's own read — the singular
values standing above a noise floor derived from the data rather than assumed. See
[`research/PAPER.pdf`](https://github.com/Agience/entroptics-context/blob/main/research/PAPER.pdf) for the
citation and the construction.

Security issues: email connect@agience.ai rather than opening a public issue.

AGPL-3.0-only, or a commercial licence — see [`LICENSE`](LICENSE), [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md),
[`NOTICE`](NOTICE) and [`CLA.md`](CLA.md).

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.
