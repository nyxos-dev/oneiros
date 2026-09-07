# Oneiros — an independent generative AI for NyxOS

**Private. Local-only. No public advances until it is stable and genuinely a
working AI.** Named for Oneiros, a child of Nyx — a model that dreams NyxOS code.

## Goal

A generative neural network, built from scratch (no ML frameworks — its own
neurons), whose purpose is to **generate N / NyxOS code from within NyxOS**.
Endgame: the model runs *inside* the OS and helps write the OS.

## Non-negotiables

- **Independent**: no external ML library, no wrapping a hosted API. The network,
  training and inference are all our own code.
- **Private**: nothing is pushed or published until it is stable and real. This
  repo has **no git remote**. Notify the maintainer *before* anything goes public.
- **Stays out of the shared trees**: this lives outside `nyx-os/`. It never edits
  `nyx-os` or Fable's `lang/` toolchain. The NyxOS source is read *only* as a
  training corpus.

## Iterations (each ~2h wake continues here and updates the log)

- **I1 — seed (DONE, this wake).** Char-level neural LM in pure C: byte embeddings
  → tanh hidden → softmax, hand-written forward/backprop, Adagrad. Trains on the
  36 MB NyxOS C corpus, checkpoints to `data/oneiros.bin`, samples C-like text.
  Proves the neurons learn (loss 5.54 → <1.6 fast).
- **I2 — signal & scale.** Held-out validation loss (train/val split) so we
  measure real generalisation, not memorisation; minibatching + a faster output
  layer; larger context/hidden. Sample across the *whole* corpus, not a slice.
- **I3 — tokeniser.** A byte-pair / word-piece vocab over C+N tokens so the model
  reasons in identifiers and keywords, not raw bytes. Big quality jump per param.
- **I4 — attention.** Replace the fixed context MLP with a small self-attention
  block (a minimal transformer, still from scratch). Longer effective context.
- **I5 — N-focused corpus.** Fold in N / N++ once Fable's examples are public API;
  teach it the language it must generate. Task-shaped prompts (signature → body).
- **I6 — fixed-point inference in NyxOS.** Quantise weights to integers, write a
  tiny inference core with no float/SSE so it runs in the ring-0 kernel budget;
  expose a `nyxgen` command that generates code inside the OS.
- **I7 — evaluation & stability gate.** Compile-rate of generated snippets via the
  in-OS `cc`; only when it reliably emits code that *builds* do we consider going
  public — and we ask first.

## Layout

    src/oneiros.c     the model (train + sample + checkpoint)
    data/corpus.txt   NyxOS C source, regenerated from nyx-os (gitignored)
    data/oneiros.bin  latest checkpoint (gitignored)
    logs/             training logs (gitignored)

Only source and docs are committed. Corpus, checkpoints and logs stay untracked.

## Log

- **I1** — model + training loop written and validated. 1M-step full-corpus run
  converged to loss **0.776**; the model emits C indentation, `static void`,
  comments, `if (`, `{ }`, `= 0;`, `++)`, `!=`, `&&`. **Key finding:** the raw
  corpus is **84.5 % hex-table bytes** (font bitmaps / embedded assets / KAT
  vectors), so the model over-learns `0xNN,` runs. Fix = train on code logic only.
- **I2 (DONE)** — cleaned corpus (strip `0x..,` hex tables): 36 MB → 5.69 MB of
  real C. Added a 90/10 **train/val split** with held-out validation loss, and
  **minibatching** (BATCH=32): the old per-sample `memset(grad)` was a hidden
  cost, so batching gave a **~3× speedup** (~3.3k → ~9.4k samp/s). Made E/B/HID
  overridable (`-DB= -DHID=`). 2M-step baseline (B=8, HID=256): train 1.86 /
  **val 1.97**, still trending down (not plateaued).
  A 600k capacity sweep {(8,256),(16,256),(16,512),(24,512)} had the *smallest*
  config best — but that was **budget-confounded** (bigger nets never converged in
  600k steps), so it does NOT prove scale won't help; a fair test needs equal-
  convergence budgets. Low-temp samples emit real C tokens (`const`, `uint8_t*`,
  `sizeof(`, `return`, `= ;`) but loop ("the the the") and lack structure → the
  char-8-context MLP captures *local statistics, not code structure*.
- **NEXT (I3) — tokeniser.** The clear lever: reason in C tokens (identifiers,
  keywords, punctuation), not raw bytes, so each step carries structure. Then a
  fair capacity re-test at equal-convergence budget.
