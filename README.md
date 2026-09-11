# Oneiros

A generative neural network for NyxOS, **written from scratch in C** — no ML
frameworks, no external libraries, no wrapping a hosted API. Its own embeddings,
its own tokenizer, its own transformer, its own backprop, even its own `expf`.
It is trained on NyxOS's own source code and it runs **inside NyxOS**.

Named for [Oneiros](https://en.wikipedia.org/wiki/Oneiroi), a child of Nyx — a
model that dreams NyxOS code.

## Honest scope

Oneiros is a **tech demo of a real from-scratch model**, not a coding tool. It
learns the *shape* of NyxOS's C — identifiers, control flow, the `nyx_*` idioms —
and generates code in that style:

```c
static int nyx_u8 d2 = (st - 1) - 2 * 3;
if ((k == 2)) { st[i]; nyx_i64 r = np;
    if (ck(st, 40)) {
```

It does **not** write code that compiles. That is a limit of scale (a small model,
a few million tokens, trained on a CPU), not a bug. What's interesting here is that
every neuron is home-grown and the whole thing runs in the OS it was trained on.

## What's in the box

The model was grown one honest, measured step at a time. Held-out loss, in
bits-per-byte (lower is better), on NyxOS code it never saw during training:

| step | model | bits/byte |
|------|-------|-----------|
| char-level MLP        | Bengio-style neural LM        | 2.84 |
| + BPE tokenizer       | reason in tokens, not bytes   | 2.54 |
| + bigger vocab + data |                               | 2.14 |
| transformer           | multi-head attn + FFN + residual | 2.01 |
| + RMSNorm             | LLaMA-style pre-norm          | 1.90 |
| + longer context (B=32) | **the champion**            | **1.78** |

Every network's hand-written backprop is checked against numerical gradients
(`<binary> gradcheck`) before it is trusted.

## Layout

    src/oneiros.c      char/token MLP (the seed)
    src/bpe.c          byte-pair-encoding tokenizer (learn / encode / decode)
    src/xformer.c      transformer block (multi-head attention + FFN + residual)
    src/xformer_ln.c   the champion: transformer + RMSNorm
    src/ngen_xln.c     self-contained inference for running INSIDE NyxOS
    model/             the trained champion + its vocabulary
    data/make_corpus.sh  builds the training corpus from a NyxOS checkout
    train.sh           reproduce a champion from scratch
    ROADMAP.md         the full iteration log; INTEGRATION.md the in-OS wiring

## Build & run (host)

```sh
gcc -O2 -march=native -DV=2048 -DB=32 -o xformer_ln src/xformer_ln.c -lm
./xformer_ln sample model/xformer_ln32.bin model/vocab2048.bin 80 0.7 "static void "
```

To train your own from a NyxOS checkout beside this one:

```sh
bash train.sh            # regenerates the corpus, learns the vocab, trains
```

## Running inside NyxOS

`src/ngen_xln.c` is the inference path built for the OS. NyxOS userland provides
`malloc`/`fopen`/`fread` and float math, but **not** `exp`/`log`/`tanh` (and the
kernel is `-mno-sse`), so Oneiros brings its own — it links with **no `-lm`** and
its output is bit-for-bit identical to the libm build under greedy decoding.

```sh
gcc -O2 -DV=2048 -DB=32 -o ngen_xln src/ngen_xln.c   # note: no -lm
```

Inside NyxOS it is the `nyxgen` command. See `INTEGRATION.md`.

## License

GPL-2.0-or-later, matching NyxOS. See `LICENSE`.
