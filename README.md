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

## Prior art

*(Add citations here after verifying them — see the note in the commit history.)*

## License

MIT