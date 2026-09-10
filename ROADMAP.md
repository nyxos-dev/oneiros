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
- **I3 (DONE)** — from-scratch byte-level **BPE tokeniser** (`src/bpe.c`): learns
  merges on the clean corpus, vocab 1024, **2.42 bytes/token**, verified **lossless
  round-trip** (decode == corpus). Learned tokens are real C structure (`int`,
  `if`, `//`, `->`, `;\n    `). Unified the model on a `sym_t` type so it trains on
  bytes (V=256) or tokens (`-DV=1024`) from a `.tok` file; token-mode sampling
  encodes the seed + decodes output via the vocab. Token model, 2M steps: val 4.26
  nats/token = **2.54 bits/byte vs the char model's 2.84 — 10.6 % better** on the
  comparable per-byte metric, and it crossed the char bpb by ~120k steps (~16×
  less training). Decoded samples emit C idioms as units (`for (int i=...; ... <
  SCREEN_WIDTH; ...)`, `out[ti] = 0;`). Train/val gap widened (3.67 / 4.26) → the
  1024-token corpus is a smaller effective dataset; mild overfitting begins.
- **I4 (DONE — honest negative)** — built a from-scratch single-head self-attention
  model (`src/attn.c`): token + positional embeddings, Q/K/V, scaled-dot-product
  attention (single query at the last position) → context → tanh head → softmax,
  hand-derived backprop **verified by a numeric gradient check** (V=256: 96.5 % of
  informative params within 5 %, mean rel err 0.013). Trained 2M steps on tokens:
  attn B=8 = **2.657 bits/byte**, attn B=16 = **2.628** — both **worse than the
  MLP's 2.540**. Longer context helps attention (B=16 < B=8) but the pooling
  bottleneck (all context → one E=24 vector) loses more than single-head attention
  gains over the MLP's full 192-dim concat. **The MLP token model stays champion
  (`oneiros.c`, 2.54 bpb);** `attn.c` is kept as the experimental branch.
- **I5 (DONE)** — bigger BPE vocab **2048** (2.97 bytes/token vs 2.42 at 1024). MLP
  token model, 2M steps: val 4.78 nats/token = **2.32 bits/byte vs 2.54 at vocab
  1024 — 8.6 % better** on the per-byte metric. **NEW CHAMPION**
  (`data/oneiros_2048.bin`, build `-DV=2048`; `train.sh` reproduces it). Decoded
  samples emit NyxOS-style signatures as units (`void nyx_start_menu_open(...)`,
  `fb_rgb(...)`, `static const int`). Train/val gap widened (3.96 / 4.86) → bigger
  vocab = fewer tokens = smaller effective dataset, so we're near the vocab ceiling
  for a 5.7 MB corpus; the next lever is MORE DATA, not just more vocab.
- **I5b (DONE — more data)** — widened the corpus to all `kernel/` + `user/` C
  (excluding Fable's `user/pkg/ncc`; adds NyxOS user code + the vendored TinyCC),
  5.7 MB → **7.6 MB clean (+33 %)**. Retrained the vocab-2048 MLP, 2M steps: val
  4.23 nats/token = **2.14 bits/byte** (from 2.32), and the **train/val gap
  collapsed 0.90 → 0.25** — more data cut overfitting ~72 %, confirming data was the
  bottleneck. NEW CHAMPION (`data/oneiros_2048_wide.bin`; `train.sh` reproduces it
  via the widened `make_corpus.sh`). Val still trending down → more steps/data
  would help further. Samples show real C constructs (`int sprite = 1;`,
  `floor->speed = ...`, `#define ...`, `while (...)`).
  bpb progression: char 2.84 → tok1024 2.54 → tok2048 2.32 → **+data 2.14**.
- **Champion pushed to 4M steps** — val 4.10 nats/token = **2.08 bits/byte** (from
  2.14 at 2M; val still slowly trending down). bpb: char 2.84 → tok1024 2.54 →
  tok2048 2.32 → +data 2.14 → +steps **2.08**.
- **I6a (DONE — in-OS-ready inference)** — `src/ngen.c`: self-contained inference
  (loads champion + BPE vocab, generates) using only malloc/fopen/fread — which
  NyxOS userland has — plus its OWN `expf`/`logf`/`tanhf` (NyxOS libm has
  sinf/cosf/sqrtf but NOT exp/log/tanh; kernel is -mno-sse). **Links with NO -lm,
  and verified byte-identical to the libm `oneiros sample`** at low temp. This is
  the inference ready to run inside NyxOS. NOTE: the actual in-OS run touches the
  nyx-os tree and is a PUBLIC step → needs the maintainer's OK first.
- **I6b (DONE — integration package ready)** — verified `ngen.c` builds under NyxOS
  constraints: **every libc symbol it uses is in `user/libc.h`** (time, atoi/atof,
  str/mem, fopen/fread/fwrite/fputc/fputs/fprintf, malloc/free), **no `-lm`**,
  conservative C99 (TinyCC-clean, no host tcc available to run the final parse).
  `INTEGRATION.md` documents the exact public wiring (drop `ngen.c` as `nyxgen`,
  put the 2.5 MB ckpt + 20 KB vocab on the disk, boot QEMU, screendump).
- **READINESS BAR** (my call; then STOP + notify — never publish myself): (1) stable
  ✓; (2) NyxOS-style output ✓; (3) self-contained NyxOS-buildable inference ✓
  (ngen.c, symbol-verified, own-math byte-identical to libm); (4a) in-OS integration
  package ✓ (`INTEGRATION.md` + data files); (4b) **quality as good as it gets at
  this scale — IN PROGRESS** (6M-step run; one honest multi-head-transformer attempt
  still owed). When (4b) settles → stop the loop + ask before the in-OS/public step.
- **6M champion = 2.04 bits/byte** (val 4.02). Gains: 2→4M −0.064, 4→6M −0.043 →
  **val plateauing, near the MLP quality ceiling at this data scale**. In-OS `ngen`
  sample: `static int nyx_i64 n, st[57])  const char* tok = 0x2000000;  #define
  OB_DER_SIZE_PROC_RELWIN_W` — real declarations/arrays/macros/comments in NyxOS
  caps style.
- **I7 (DONE — the transformer WINS)** — full transformer block from scratch
  (`src/xformer.c`): multi-head attention (NH=4) + FFN + residual + a matched
  D→HID→V head, D=64; hand backprop **gradient-checked** (V=256: 100 % within 5 %).
  First run DIVERGED (loss 18 > random — no LayerNorm + Adagrad cold-start); fixed
  with LR **warmup + lr 0.02** (env `ONEIROS_LR`). 4M steps: val 3.96 = **2.01
  bits/byte — NEW CHAMPION**, beating the 6M MLP (2.04) at fewer steps and the 4M
  MLP (2.08) clearly. **Honest correction: the MLP was NOT the ceiling — I called it
  too early; the proper transformer opened fresh headroom** (val still ~flat-dropping
  at 4M). Sample: `static int nyx_i64 fl = nyx_u64 + (nyx_str){...}; tag =
  ptcc_read(); if(!`.
- **I7b (DONE — transformer scales + runs in-OS)** — resumed +4M (8M total): val
  3.76 = **1.91 bits/byte** (from 2.01 at 4M); still improving. And `src/ngen_x.c` =
  the transformer's self-contained in-OS inference (own expf/logf/tanhf + own sqrtf,
  no `-lm`), **verified byte-identical to the libm `xformer`**. So the CHAMPION now
  runs under NyxOS constraints, not just the old MLP. In-OS sample: `int main()
  return (int)(gif_y1); #define R_PPC_RELLIST 7`.
- **I7c (DONE — RMSNorm helps)** — RMSNorm (LLaMA-style pre-norm) at 3 points
  (`src/xformer_ln.c`), gradient-checked (V=256 97 %; V=2048 100 % within 5 % on
  |grad|>1e-2, rest is float noise). Trains stably at **lr 0.05** (2.5× the no-norm
  lr) and hits **1.90 bits/byte at 4M steps — NEW CHAMPION**, beating the no-norm
  transformer's 1.91 at 8M (better in HALF the steps). Still dropping. Sample:
  `void nyx_i64* pct_lock, nb_section->data, fmt) return 1; }`.
- **I7d (DONE — longer context + in-OS champion)** — B=8→**B=16** (RMSNorm handles
  it; attention params barely grow with B): val 3.59 = **1.82 bits/byte at 4M — NEW
  CHAMPION** (from 1.90 at B=8), still dropping. And `src/ngen_xln.c` = the RMSNorm
  transformer's self-contained in-OS inference (own math incl. own sqrtf, no `-lm`),
  **verified byte-identical to libm** at B=8 and B=16 → the in-OS demo now runs the
  exact champion. In-OS sample: `static int nyx_u8 d2 = (st-1)-2*3; if ((k==2)) {
  st[...]; nyx_i64 r = np; if (ck(st,40)) {` — real conditionals, indexing, nesting.
- bpb: char 2.84 → tok1024 2.54 → tok2048 2.32 → +data 2.14 → MLP 2.04 → transformer
  2.01 → 1.91 → RMSNorm 1.90 → **B=16 1.82**.
- **NEXT** — still improving: even longer context (B=32), more steps, or 2 blocks.
  **(4b) not settled, not preparada.**
- Loop continues (2h cron). Champion = `xformer_ln16.bin` (RMSNorm B=16); in-OS =
  `ngen_xln.c -DB=16`. Private; the in-OS run needs the user's OK.
