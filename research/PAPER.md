# Selection for language-model context, read from the data

### Using entroptics to decide how many retrieved items to keep, which to keep, and where foreign material sits

**Ikailo John Sessford**, Ikailo Inc., `john@ikailo.com`

*Version 0.2.1. November 2026.*

---

## Abstract

A language model's context is assembled from several sources at once: a retrieval-augmented
prompt, an agent scratchpad, a long conversation. The decisions made about it — how many
retrieved items to keep, which of them to trust, whether an answer stayed on its evidence — are
conventionally made with one number per item against one summary of the whole: a cosine, a
centroid, a fixed top-$k$.

This work reads the context as the $K$ directions it resolves above its own noise floor, using
the `entroptics` instrument, and makes each of those decisions from that span. $K$ is counted
from the data, so nothing is trained, no threshold is set, and no constant is fitted per corpus.

**Where relevance spreads across several facets, the derived cut holds $F_1$ at 1.000 through
four facets and at 0.915 and 0.711 at five and six, where a cosine cut falls to 0.000.** On
locating a real prompt injection planted in real documents, it names the attack as the first item
to inspect more than four times as often as a centroid does at five sources, and leads a
k-nearest-neighbour detector outright at one and two sources. On deciding
whether a continuation belongs to its context, it ties a centroid at one source and leads it
by a margin that grows from two sources to five.

One statement covers all of it. **These reads beat a single summary statistic exactly when the
deciding distribution is multi-modal, and the margin grows with how multi-modal it is.** Where
the deciding distribution has one mode they reduce to the statistic they are measured against
and tie; on four BEIR corpora and an end-to-end HotpotQA run they tie with cosine.

---

## 1. The claim

> **A read taken over the $K$ directions a context resolves decides better than one taken
> against a single summary of it, exactly when the deciding distribution is multi-modal, and
> the advantage grows with how multi-modal it is.**

Three measurements carry it, on three different axes of modality: facets of a query's relevance
(§4), sources in a context (§5), and topics spanned by a conversation (§6). On two real-corpus
measurements the reads tie with cosine (§7).

## 2. A span in place of a point

Take a context $C = \{x_1, \dots, x_n\}$ of direction-normalized embeddings drawn from $M$
distinct sources. The centroid

$$\bar{x} = \frac{1}{n}\sum_i x_i \Big/ \Big\lVert \frac{1}{n}\sum_i x_i \Big\rVert$$

is a single point on the sphere, and ranking by cosine to it is identical to ranking by average
similarity to the rest of the set — the standard summary an item is usually scored against.
Where the sources occupy $M$ well-separated directions, $\bar{x}$ is their weighted mean and for
$M > 1$ it points into the gap between them. The cosine of a legitimate item from source $m$
then falls roughly as $\cos\theta$, with $\theta$ that item's angle from the consensus
direction — a quantity set by **how spread the context is**, carrying no information about the
item.

A subspace read puts a span where that point was. Let $X \in \mathbb{R}^{n \times d}$ stack the
context, let $X = U\Sigma V^\top$, and take

$$V_K = [v_1, \dots, v_K], \qquad K = \#\{\,\sigma_j : \sigma_j \text{ resolves above the noise floor}\,\}.$$

Score an item by the energy it retains under projection:

$$\phi_{\text{in}}(y) = \frac{\lVert V_K^\top y \rVert^2}{\lVert y \rVert^2} \in [0, 1].$$

Each resolved source direction lies in $\operatorname{span}(V_K)$, so a legitimate item from
*any* source scores high: its own direction is represented whether or not the others are. The
span admits every source at once, and the margin over a single-point summary widens as the
sources multiply, because the point moves further from all of them while the span keeps holding
each.

## 3. Choosing $K$ without choosing anything

$K$ is the one quantity that could be tuned, and it is read rather than set. It comes from the
singular spectrum via the `entroptics` instrument [1], which counts the modes standing above the
noise floor derived from the block's own shape. For an $n \times d$ block of pure noise the
singular values follow a Marchenko-Pastur law with a Tracy-Widom edge; modes above that edge are
the ones the block resolves. The count is a property of the data and moves with the context: in
§5 it reads 5.6 modes at one source and 12.6 at five, without being told how many sources there
were.

The modes are taken off the block **as recorded**, column means included. A constant level is a
mode: in the optics in which the library speaks it is the zero-order beam, physically present and passed
by the aperture like any other. For a set of embeddings that matters directly, because **the
shared mean direction is the topic the set is about** — it is the leading coherent mode, and it
carries signal. The count is read on the library's whitened frame, where the noise floor is
defined, and the whitening centres each channel on its mean, so it counts the directions about
the shared mean and not the mean itself. Where the mean is a mode of its own the count is one
short of the block's modes; Appendix A.1 shows the anomaly reads under that count.

Absence is handled by masking. A row or column that was never measured is dropped before the
decomposition and never reaches the denominator. A channel of measured zeros is an observation
of no power and counts as a mode. The two are distinct inputs and give distinct values of $K$,
which is what a caller with sparse features needs to know.

## 4. How many to keep, when relevance has several facets

The largest margin in this work is on selection. A query whose relevance is spread across $R$
independent facets is the multi-modal case at its cleanest, and $R$ is a controlled variable:
the facets are planted, so the axis is exact rather than estimated.

Both sides run the same parameter-free lock over the same head screen; the only difference is
whether the screen precedes the lock (`research/benchmarks/bench_facets.py`, 60 items, 12 gold,
16 heads, 40 draws).

| facets $R$ | cosine $F_1$ | **entroptics** $F_1$ | edge | 95% CI |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 2 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 3 | 1.000 | 1.000 | $+0.000$ | [$+0.000$, $+0.000$] |
| 4 | 0.806 | **1.000** | $\mathbf{+0.194}$ | [$+0.099$, $+0.301$] |
| 5 | 0.000 | **0.915** | $\mathbf{+0.915}$ | [$+0.866$, $+0.955$] |
| 6 | 0.000 | **0.711** | $\mathbf{+0.711}$ | [$+0.624$, $+0.793$] |

**Entroptics holds $F_1$ at 1.000 through four facets and keeps 0.915 and 0.711 at five and
six.** The cosine cut matches it while the relevance has one to three facets and falls to 0.000
by five, so the edge opens at four facets and is largest at five. The reason is §2's, transposed
from a context to a query: one scalar carries one facet's worth of evidence, and each gold
item's total weakens toward the distractor level as the facets multiply, where an $R$-mode
subspace holds all of them at once.

The count kept is derived per query throughout. Nothing is swept, and the same lock produces
both columns.

## 5. Which item is foreign, when a context spans several sources

The second axis of modality is the number of sources a context is assembled from. Legitimate
content is whole wikitext-103 articles, so each source is a genuine topic; the attacks are the
`deepset/prompt-injections` corpus, strings written to attack real systems. One attack is
appended to a context built from $M$ sources and the read ranks it. Encoder is
`bge-small-en-v1.5`, 200 trials per row, intervals paired per trial
(`research/benchmarks/bench_injection.py`).

Two baselines. The **centroid** is the standard single summary of §2. **k-NN** scores an item by
its similarity to its five nearest neighbours — a local read, so it carries no penalty for a
context that spans topics, and it is the stronger of the two.

| sources | modes | items | **entroptics** | centroid | k-NN | vs centroid | vs k-NN |
|---|---|---|---|---|---|---|---|
| 1 | 5.6 | 39 | **0.991** | 0.983 | 0.984 | $+0.007$ [$+0.004$, $+0.012$] | $\mathbf{+0.007}$ [$+0.004$, $+0.010$] |
| 2 | 7.7 | 77 | **0.969** | 0.902 | 0.959 | $+0.067$ [$+0.053$, $+0.082$] | $\mathbf{+0.010}$ [$+0.005$, $+0.016$] |
| 3 | 9.6 | 116 | **0.952** | 0.833 | 0.947 | $+0.119$ [$+0.098$, $+0.140$] | $+0.004$ [$-0.003$, $+0.012$] |
| 5 | 12.6 | 191 | 0.897 | 0.718 | **0.917** | $+0.179$ [$+0.154$, $+0.205$] | $-0.020$ [$-0.030$, $-0.011$] |

**Against the standard summary the margin grows from $+0.007$ to $+0.179$**, every interval
excluding zero, exactly as §2 predicts: the centroid falls from 0.983 to 0.718 as the sources
multiply while entroptics holds from 0.991 to 0.897.

**Against a local detector entroptics leads where the context is tightest** — $+0.007$ and
$+0.010$ at one and two sources, both intervals excluding zero — and the order reverses as the
context widens: at five sources k-NN leads by $0.020$ [$0.011$, $0.030$].

The operator's question is which item to open first, and there entroptics has the highest rate
at every source count as point estimates, though at three sources k-NN is within $0.015$ of it:

| sources | **entroptics** | centroid | k-NN |
|---|---|---|---|
| 1 | **0.795** | 0.735 | 0.720 |
| 2 | **0.465** | 0.250 | 0.365 |
| 3 | **0.315** | 0.145 | 0.300 |
| 5 | **0.140** | 0.030 | 0.120 |

At five sources it names the attack first more than four times as often as the centroid does,
from a pool of 191 items. Each item is scored against the subspace of the
set without it; Appendix A gives the reason.

## 6. Whether a continuation belongs to its context

The third axis is a conversation's topic spread, and the question is non-adversarial: **is this
continuation on-topic?** Held-out sentences from the context's own sources — on-topic, and held
out of the context — are scored against sentences from a different article. Same encoder, 200
trials per row, six held-out sentences per source so the target count resolves every effect
reported (`research/benchmarks/bench_drift.py`).

| sources | modes | context | **entroptics** | centroid | k-NN | vs centroid |
|---|---|---|---|---|---|---|
| 1 | 2.3 | 12 | 0.961 | 0.960 | **0.966** | $+0.000$ [$-0.002$, $+0.003$] |
| 2 | 3.1 | 24 | 0.933 | 0.890 | **0.948** | $\mathbf{+0.043}$ [$+0.037$, $+0.051$] |
| 3 | 4.1 | 36 | 0.915 | 0.827 | **0.936** | $\mathbf{+0.088}$ [$+0.077$, $+0.098$] |
| 5 | 6.0 | 60 | 0.879 | 0.738 | **0.909** | $\mathbf{+0.141}$ [$+0.130$, $+0.153$] |

**The same growth, from an independent task.** The edge over the standard summary opens at two
sources and grows to $+0.141$ at five, every interval from two upward excluding zero, with the
centroid falling 0.960 → 0.738 while entroptics holds 0.961 → 0.879. §5 appends foreign text and
this withholds native text; they are measured separately and agree on how the margin grows
with the sources.

A local neighbour read leads on this task throughout, by $0.005$ to $0.030$, every interval
excluding zero. Ranking a target
against its nearest context sentences is a close fit to the question as posed, and the value
entroptics adds here is the count it derives rather than the ordering it produces — §4 and §7
measure that directly.

## 7. Real corpora, where relevance has one mode

The law predicts a tie wherever the deciding distribution has one mode, because the read
reduces to the statistic it is measured against. Two real-corpus measurements confirm it.

**Four BEIR corpora** — nfcorpus, scifact, arguana, fiqa, 2,421 scored queries, encoded with
`all-MiniLM-L6-v2` — against the identical
`top_break` lock run on plain cosine salience, so the screen is the only difference
(`research/benchmarks/bench_beir.py`):

| dataset | cosine + lock | **entroptics** | edge | 95% CI |
|---|---|---|---|---|
| nfcorpus | 0.295 | 0.291 | $-0.005$ | [$-0.018$, $+0.009$] |
| scifact | 0.571 | 0.558 | $-0.013$ | [$-0.029$, $+0.002$] |
| arguana | 0.257 | 0.256 | $-0.001$ | [$-0.007$, $+0.005$] |
| fiqa | 0.344 | 0.341 | $-0.003$ | [$-0.012$, $+0.007$] |

Four ties, every interval containing zero. At the point the cut sees them, 198 of the 2,421
scored queries (8.2%) carry four or more relevant documents in their horizon: 144 of
nfcorpus's 260, 4 of scifact's 270, none of arguana's 1,357 and 50 of fiqa's 534. On three of
the four corpora the deciding distribution has one mode and §4's condition is absent.

**HotpotQA end-to-end**, 150 questions, a language model answering from the selected context
(`research/benchmarks/bench_hotpot_e2e.py`). The decisive arm is cosine's top-$m$, where $m$ is
the count entroptics derived *for that question*, so the only difference is which items were
chosen:

| model | fixed $k=5$ | cosine top-$m$ | **entroptics** | entroptics − top-$m$ |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.490 | 0.432 | 0.432 | $+0.000$ [$+0.000$, $+0.000$] |

The selection matches cosine's, as the law requires on unimodal relevance.

**What the derived count buys where the reads tie is the count itself.** The best single
constant $k$ over the four corpora, top-1, gives up as much as $0.130$ $F_1$ against a
corpus's own best fixed $k$ (nfcorpus, whose best is top-10); reading the count per query
gives up at most $0.020$. That is the value where there is no sweep to run — a
service answering against more than one corpus, or a corpus whose content moves.

**Size the budget to the task.** On HotpotQA the derived count spends 38% of a fixed top-5's
context tokens, 224 against 593, keeping 2.1 paragraphs where a multi-hop answer needs two gold
paragraphs. Against that fixed top-5 the difference is $-0.058$ [$-0.123$, $+0.006$], and what
there is of it is the count rather than the ranking: the matched-budget arm above shows the
ranking already matches the baseline. Where a task needs several passages, set a floor on the count.

## 8. What this costs

One thin SVD of the context block, $O(n d \min(n,d))$, on embeddings the pipeline has already
computed for retrieval. No additional model call, no second encoder, no index. Scoring an item
against the basis is a $d \times K$ matrix-vector product. For the context sizes here — tens to
a few hundred items — the read is well under the cost of the retrieval that produced them.

The anomaly read of §5 scores each item against the subspace of the set without it, which is
$n$ decompositions rather than one. Appendix A.1 gives the reason, the alternative and what each
scores.

## 9. Scope

**Coherence reads topic.** An answer that stays on its sources' subject scores high whether or
not it is correct: on the 88 HotpotQA questions whose context kept two or more paragraphs,
right answers scored 0.413 and wrong ones 0.417, a separation of $-0.004$. It is a monitor for off-source and lost-thread
behaviour.

**Scores and rankings.** The reads order items within one context. "Is there an anomaly in this
set at all" is a question about a reference distribution, which a caller supplies;
`anomaly_z` takes one.

**Measured on two encoders.** The BEIR table is `all-MiniLM-L6-v2`, every other embedding
number is `bge-small-en-v1.5`, and the end-to-end table is Qwen2.5-1.5B. The mechanism in §2 is a statement about
direction-normalized embeddings generally; the magnitudes are measured on these models, and $K$
depends on how a model spreads a topic.

## 10. Reproducing

```bash
pip install -e ".[bench]"
python -m research.benchmarks.bench_facets       # §4
python -m research.benchmarks.bench_injection    # §5
python -m research.benchmarks.bench_drift        # §6
python -m research.benchmarks.bench_beir --datasets nfcorpus scifact arguana fiqa   # §7
```

The end-to-end table in §7 needs a language model as well:

```bash
pip install -e ".[bench,bench-llm]"
python -m research.benchmarks.bench_hotpot_e2e --n 150
```

`bench_facets` runs on CPU in seconds and needs no corpus. The others download their data from
Hugging Face. Every number in this paper carries the conditions it was measured under.

---

## Appendix A. Two choices in the API's shape

**A.1 — Each item is scored against the subspace of the set without it.** `anomaly` leaves the
item out before building the span, so the span an item is scored against never contains a mode
that item supplied. That is a structural argument, and the default rests on it alone. The
alternative, `leave_one_out=False`, builds one span from every item: one decomposition instead of
$n$, with the item scored inside a span it helped build.

What the two reads score on clean synthetic sets: one injected unit item with amplitude $f$ on a
direction no topic uses and $\sqrt{1 - f^2}$ on one topic; eight items per topic, noise norm
0.25, $d = 192$, 15 draws, top-1 localization (`research/benchmarks/bench_leave_one_out.py`):

| foreignness $f$ | read | 1 topic | 3 topics | 5 topics |
|---|---|---|---|---|
| 0.7 | in-set | 1.00 | 0.00 | 0.27 |
| 0.7 | leave-one-out | 1.00 | 0.00 | 0.00 |
| 0.9 | in-set | 1.00 | 0.93 | 1.00 |
| 0.9 | leave-one-out | 1.00 | 0.80 | 0.00 |
| 1.0 | in-set | 1.00 | 1.00 | 1.00 |
| 1.0 | leave-one-out | 1.00 | 1.00 | 1.00 |

The count is read on the whitened frame, which centres each channel on its mean (§3), and on
the clean set it is one short of the topics from three up: two at three topics, four at five,
one at one. The same benchmark prints, per cell, the injected item's anomaly and the highest
anomaly among the other items, means over the draws:

| $f$ | topics | in-set: item | in-set: other | leave-one-out: item | leave-one-out: other |
|---|---|---|---|---|---|
| 0.7 | 3 | 0.43 | 0.66 | 0.65 | 1.00 |
| 0.7 | 5 | 0.44 | 0.51 | 0.61 | 0.99 |
| 0.9 | 3 | 0.77 | 0.20 | 0.87 | 0.27 |
| 0.9 | 5 | 0.78 | 0.49 | 0.86 | 0.99 |
| 1.0 | 3 | 1.00 | 0.06 | 1.00 | 0.08 |
| 1.0 | 5 | 1.00 | 0.46 | 1.00 | 0.99 |

At one topic the other items read 0.06 to 0.08 under both reads. In a draw the injected item is found
where its score clears the highest among the rest. Under leave-one-out some other item reads 0.99 to 1.00
outside the span built without it at five topics for every $f$, and at three topics at $f = 0.7$;
under the in-set read the highest other item reads 0.46 to 0.66 in those cells. In this
construction the in-set read's top-1 rate is at or above leave-one-out's in every cell, though
the injected item itself scores higher under leave-one-out, so the table does not support the
default. The §5 figures are measured with the leave-one-out read and this count, and
stand as measured.

**A.2 — A subspace overlap is read against its null.** In high dimension two subspaces that are
not specifically aligned are near-orthogonal, so a raw `subspace_overlap` reports "most of the
structure lies outside" for a random matrix as readily as for a real one, measuring the ambient
dimension. The analytic null is $k_b/d$ — a random $k_b$-dimensional subspace covers that
fraction of anything — and `chance_overlap` returns it. Report the ratio of measured overlap to
chance.

---

## References

[1] **entroptics** — the spectral instrument. Resolved-mode counting against a
Marchenko-Pastur/Tracy-Widom noise floor, and the occupancy read used here to set $K$.
Apache-2.0. <https://pypi.org/project/entroptics/>

[2] **wikitext-103** — Merity et al., *Pointer Sentinel Mixture Models*, ICLR 2017.

[3] **deepset/prompt-injections** — a corpus of real prompt-injection strings.
<https://huggingface.co/datasets/deepset/prompt-injections>

[4] **BEIR** — Thakur et al., *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of
Information Retrieval Models*, NeurIPS 2021.

[5] **HotpotQA** — Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
Answering*, EMNLP 2018.

## Citation

```bibtex
@techreport{ikailo2026entropticscontext,
  title  = {Selection for language-model context, read from the data},
  author = {{Ikailo Inc.}},
  year   = {2026},
  note   = {https://github.com/Agience/entroptics-context}
}
```
