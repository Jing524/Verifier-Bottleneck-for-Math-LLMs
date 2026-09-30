# Verifier Bottleneck for Math LLMs

> **Blog:** [Is the Next Sample Worth the Read?](https://jing524.github.io/Verifier-Bottleneck-for-Math-LLMs/) ·
> **Source:** [blog branch](https://github.com/Jing524/Verifier-Bottleneck-for-Math-LLMs/tree/blog)

This repository studies whether additional **test-time sampling** for mathematical reasoning models improves accuracy fast enough to justify the extra reasoning tokens.

The project focuses on three related questions:

- **Generation:** does a correct answer appear in the candidate pool?
- **Selection:** if a correct candidate exists, can voting or a verifier identify it?
- **Utilization:** is the resulting accuracy gain worth the additional inference cost?

The accompanying research note proposes **Structural Task Utilization (STU)**:

$$ \mathrm{STU} = \frac{A\beta\rho}{\alpha\tau} $$

where $\tau=T/1024$ is the normalized token cost.

---

## Key Findings

- On **AIME 2025**, one sample from Qwen3-4B-Thinking-2507 solves **25/30 problems (83.3%)**.
- An **8-sample majority vote** solves **27/30 problems (90.0%)**.
- Accuracy increases by only **1.08×**, while tokens per problem increase by **8.15×**.
- Consequently, $A/\tau$ at 8 samples is only **13.2%** of the one-sample value.
- On a diagnostic medium-difficulty set, coverage rises from **6/12 to 12/12**, while $A/\tau$ falls from **0.136 to 0.017**.
- Three AIME problems remain unsolved after **32 samples each**, with repeated wrong answers forming stable error modes.
- A verifier does not always recover correct answers that are already present in the candidate pool.

> **Main observation:** More test-time compute can improve capability without necessarily improving utilization.

---

## Structural Task Utilization

| Symbol | Meaning |
| --- | --- |
| $A$ | Exact-answer accuracy |
| $\beta$ | Attention assigned to answer-relevant tokens |
| $\rho$ | Fraction of read tokens relevant to the answer |
| $\alpha$ | Activated-parameter ratio |
| $\tau$ | Normalized token cost, $\tau=T/1024$ |

The current experiments directly measure $A$ and $\tau$.

The factors $\beta$ and $\rho$ are **not yet directly measured**. Since Qwen3-4B-Thinking-2507 is dense, $\alpha=1$.

Therefore, the empirical score reported in the current experiments is primarily:

$$ \frac{A}{\tau} $$

rather than full STU.

---

## Pass@n and Vote@n

**Pass@n** measures whether at least one of the first $n$ samples contains the correct answer.

**Vote@n** measures whether majority voting over those $n$ samples returns the correct answer.

For example, on AIME 2025 at $n=2$:

- **Pass@2 = 26:** 26 of 30 problems have at least one correct candidate.
- **Vote@2 = 24:** majority voting returns the correct answer on only 24 of 30 problems.

The gap between the two reflects a **selection bottleneck**: generating a correct candidate does not guarantee that it will be selected.

---

## AIME 2025 Results

| Width $n$ | Pass@n | Vote@n | Tokens / problem | $A/\tau$ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25 | 25 | 22,247 | 0.0384 |
| 2 | 26 | 24 | 43,533 | 0.0188 |
| 3 | 26 | 25 | 65,489 | 0.0130 |
| 4 | 26 | 26 | 88,859 | 0.0100 |
| 5 | 26 | 26 | 112,115 | 0.0079 |
| 6 | 27 | 26 | 134,349 | 0.0066 |
| 7 | 27 | 26 | 157,533 | 0.0056 |
| 8 | 27 | 27 | 181,335 | 0.0051 |

From one sample to eight samples:

| Quantity | Change |
| --- | ---: |
| Accuracy | 83.3% → 90.0% |
| Accuracy ratio | 1.08× |
| Token ratio | 8.15× |
| Relative $A/\tau$ | 0.132× |

The wider pool improves accuracy, but the read cost grows much faster than the accuracy gain.

The full analysis, including verifier behavior, reflection, answer-position checks, and stable error modes, is described in the [research note](https://jing524.github.io/Verifier-Bottleneck-for-Math-LLMs/).

---

## Experiments

The repository contains several complementary experiments:

- **Short-budget test** — tests whether a single reasoning trace benefits from a larger generation allowance.
- **Parallel-width scaling** — compares sampling widths $n=1,2,4,8,16$.
- **Verifier selection** — tests whether a verifier can recover correct candidates already present in the pool.
- **Reflection** — compares critique-and-rewrite against token-matched independent resampling.
- **Answer-position analysis** — examines where the final boxed answer appears in the reasoning trace.
- **AIME 2025** — compares one sample with an 8-sample majority vote under contest-style sampling.
- **Stable error modes** — extends three persistent AIME failures from 8 to 32 samples.

---

## Experimental Setup

Main model:

`Qwen3-4B-Thinking-2507`

AIME 2025 sampling:

- temperature: `0.6`
- top-p: `0.95`
- top-k: `20`
- maximum new tokens: `81920`

Serving:

- vLLM
- 8 GPUs
- one server per GPU

Token cost is defined as:

$$ T = \text{prompt tokens} + \text{solution completion tokens} $$

with:

$$ \tau = \frac{T}{1024} $$

Verifier inference tokens are currently not included in $\tau$.

---

## Repository Layout

| Path | Role |
| --- | --- |
| `experiments/metrics.py` | Voting, token accounting, $A/\tau$, and bootstrap intervals |
| `experiments/analyze.py` | Rebuilds the AIME tables and answer-position analysis |
| `experiments/run_aime_full.py` | Main AIME 2025 experiment |
| `experiments/run_small.py` | Width scaling, verifier ratings, and reflection |
| `experiments/run_enrich.py` | Short generation-budget and longer-critique experiments |
| `experiments/run_support.py` | Additional samples for persistent AIME failures |
| `experiments/start_vllm.sh` | Starts one vLLM server per GPU |

See [`experiments/README.md`](experiments/README.md) for the detailed experimental runbook.

The `legacy/` directory contains earlier runners that are not part of the final reported protocol.

---

## Reproducing the AIME Analysis

Start the vLLM servers:

```bash
bash experiments/start_vllm.sh
```

Run the full AIME experiment:

```bash
python3 experiments/run_aime_full.py
```

Recompute the analysis:

```bash
python3 experiments/analyze.py
```

The analysis checks the headline majority-vote counts:

- $n=1$: 25
- $n=2$: 24
- $n=4$: 26
- $n=8$: 27

### Raw Traces

Raw generation traces are not currently included in the repository.

`experiments/analyze.py` expects them locally under:

```text
experiments/results/
```

Full trace-level reproduction therefore requires rerunning the model.

---

## Limitations

- The main experiments use a single reasoning model.
- AIME 2025 contains only 30 problems.
- $\beta$ and $\rho$ are not directly measured, so the current experiments report $A/\tau$ rather than full STU.
- Verifier inference cost is not included in $\tau$.
- The medium set and reflection experiment are diagnostic, small-scale studies.

---

## Research Note

For the complete derivation, experimental analysis, figures, and discussion:

**[Is the Next Sample Worth the Read?](https://jing524.github.io/Verifier-Bottleneck-for-Math-LLMs/)**

The source is maintained on the `blog` branch:

```bash
git checkout blog
python3 build_site.py
```

Then open `index.html`.

---

## License

MIT License.
