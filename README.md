# The Ramanujan Oracle

**A neural network with 8,192 hidden units whose entire trained state is 42 numbers.**

## The result

| Metric | Value |
|---|---|
| Hidden units | 8,192 |
| Weights synthesized per forward pass | 24,576 |
| Learnable parameters | **42** |
| Accuracy on XOR | 100% |
| Model file | 670 bytes (42 numbers) |
| Persistence test | bit-identical reload |
| Equivalent standard MLP at same width | 32,769 trainable parameters |
| Compression at this width | 780.2× |

## What this is

A working, runnable implementation of the **Zero-Map principle**: weights can be
computed from a formula rather than stored on disk.

The 24,576 weights of this network do not exist as a file. They are synthesized
on every forward pass from the digits of 1/π using Ramanujan's 1914 series:

    1/π = (2√2 / 9801) · Σₖ (4k)!(1103 + 26390k) / ((k!)⁴ · 396^{4k})

The network learns only:

- 10 digit-embedding values per layer (what each digit 0–9 contributes)
- 10 bias-embedding values per layer
- 1 scale per layer

= **42 numbers total.**

## What this is NOT

- **Not** a claim that XOR requires 32,769 parameters. A 3-hidden-unit MLP solves XOR with 16 parameters.
- **Not** a transformer replacement.
- **Not** a general-purpose compression scheme.
- **Not** a claim of novel invention. The Zero-Map principle has been discussed in recent preprints. This repo is an **independent, from-scratch, runnable instance** of the principle — with a working model file, a passing persistence test, and a full debugging log.

## What this IS

A reproducible demonstration that **width stops costing learnable parameters**
when weights are synthesized from a formula. Compression ratio grows linearly
with width:

| hidden | learnable | synthesized | compression |
|---|---|---|---|
| 2048 | 42 | 6,144 | 146.3× |
| 4096 | 42 | 12,288 | 292.6× |
| 8192 | 42 | 24,576 | **780.2×** |

## Reproduce

    pip install torch
    python ramanujan_oracle.py      # trains, saves, reloads, fair-baseline compares
    python ramanujan_inference.py   # loads the 670-byte file and runs 100% inference
    python see_it_work.py           # live demo of the mechanism

## The model file

`ramanujan_oracle.txt` (670 bytes) is the complete trained network. It contains
only the 42 learnable numbers. There is no weights file. Loading it into a fresh
process produces bit-identical outputs.

## How it works in one paragraph

The first 24,576 digits of 1/π are assigned to weight positions by index. Each
digit `0–9` maps through a learned 10-value lookup table to a real number.
Multiply by a learned scale, and you have the entire weight matrix. Because
thousands of weights share the same digit, changing **one** of the 42 learnable
numbers changes thousands of weights simultaneously. That is how 42 numbers can
steer 24,576 weights. The tradeoff is expressivity — the weight matrix can only
take 10 distinct values per layer.

## Research log (v1 → v6)

- **v1** — single scalar per layer. Collapsed to loss = ln(2).
- **v2** — digit embeddings. 100% at hidden=256. 18× compression.
- **v3** — scaled to hidden=2048 with a 1500-digit stream. Failed.
- **v4** — extended stream to 7000 digits. Still failed. Wrap hypothesis wrong.
- **v5** — added dead-neuron diagnostic. **ReLU at hidden=2048 killed 25% of neurons and collapsed gradient flow.** Leaky ReLU (slope 0.1) restored it. 100% at 42 params.
- **v6** — fair baseline sweep at hidden=8192. Confirmed 32,769 params required at matched width.

The v3/v4 failures were optimization pathologies, not capacity limits. That
distinction is the interesting part.

## Stage 1 — Conditional Zero-Map Weight Generation

One generator, conditioned on a task ID, produces working weight meta-parameters
for a family of boolean tasks. The 1/pi substrate is shared; the generator
learns how to modulate it.

### Mechanism test (N=4)

- XOR, AND, OR, 3-bit parity: all at 100%
- Generator: 418 params (task ID -> 42 meta-parameters)
- Target network: 1,024 synthesized weights per task
- Unexpected: 3-bit parity solved — the 10-distinct-values substrate is more
  expressive than expected when conditioned

### Scaling experiment

Does the generator beat N separate 42-param models as N grows?

| N tasks | generator params | N x 42 separate | ratio | solved |
|---------|------------------|-----------------|-------|--------|
| 4       | 418              | 168             | 0.40x | 4/4    |
| 10      | 466              | 420             | 0.90x | 10/10  |
| **20**  | **546**          | **840**         | **1.54x** | **20/20** |
| 50      | 786              | 2100            | 2.67x | 50/50  |
| 100     | 1186             | 4200            | 3.54x | **98/100** |

- **Crossover at N=20.** Below 20 tasks, the generator costs more than N
  separate models. Above 20, it wins, and the win grows with N.
- **Clean scaling through N=50** — 50/50 solved, generator 2.67x smaller than
  50 separate models.
- **Capacity limit at N=100** — 2 of 100 tasks failed (min accuracy 75%, mean
  99.5%). With only 8 generator hidden units, the generator cannot linearly
  separate 100 distinct one-hot inputs — a rank limit, not a bug. This is the
  honest boundary of the current substrate.

### What this is

A working demonstration that conditional zero-map weight generation scales:
one fixed-size generator produces working weight sets for a growing family of
tasks, with parameter cost sublinear in N. The generator stays under 1,200
parameters while solving up to 50 tasks perfectly.

### What this is not

- Not a transformer replacement. 100 boolean tasks is not language.
- Not a general-purpose scheme. The 1/pi substrate is fixed; the tasks are
  simple boolean functions over 3 bits.
- Not a capacity claim beyond N=100. The substrate hits its limit at
  approximately 100 tasks with 8 generator hidden units.

### Capacity test at N=100 — three generator widths

The N=100 failure at gen_h=8 (98/100) was probed by widening the generator.

| gen_h | gen_params | ratio vs 100 separate | solved | min_acc |
|-------|------------|----------------------|--------|---------|
| 8     | 1186       | 3.54x smaller        | 98/100 | 75.0%   |
| 16    | 2330       | 1.80x smaller        | **95/100** | 62.5% |
| 32    | 4618       | **0.91x — larger**   | 100/100 | 100.0% |

Three honest observations:

1. **Non-monotonic scaling.** Widening from 8 to 16 hidden units produced
   *worse* results (98 -> 95 solved). This is not a capacity ceiling — it is
   an optimization pathology in the two-layer MLP generator. More width is
   not monotonically better at this task count.
2. **"Solved" is not a win.** gen_h=32 solves 100/100, but requires 4618
   parameters — more than 100 separate 42-param models (4200). The generator
   lost the compression advantage it had at smaller gen_h.
3. **The strongest clean claim remains N=50** — 50/50 solved, generator 2.67x
   smaller than 50 separate models. N=100 is where the current architecture
   breaks down, in a way that is neither a clean capacity ceiling nor a
   substrate ceiling.

### What this is

A working demonstration that conditional zero-map weight generation scales
cleanly to N=50 (2.67x compression, 50/50 solved), with a documented breakdown
at N=100 that is architecture-specific rather than fundamental.

### What this is not

- Not a transformer replacement. 50-100 boolean tasks is not language.
- Not a capacity or substrate ceiling claim. The N=100 regression at gen_h=16
  suggests optimization pathology, not a hard limit. Not yet distinguished.
- Single-seed results. The 98 -> 95 non-monotonicity needs replication across
  seeds before being called a robust phenomenon.

### Next experiment

Seed sweep: run gen_h in {8, 16, 32} at N=100 with 3 seeds each. If
gen_h=16 is consistently worse than gen_h=8 across seeds, the non-monotonicity
is a real architectural finding. If it's seed noise, the clean N=100 result
likely needs a different generator architecture (residual connection, layer
norm, or a wider-but-shallower MLP).

See `conditional_oracle.py` (mechanism), `scaling_experiment.py` (scaling),
and `capacity_test.py` (this table).


## License

MIT