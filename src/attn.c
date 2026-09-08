/*
 * Oneiros - an independent generative neural network for NyxOS.
 *
 * I4: a minimal self-attention language model, still from scratch, no ML
 * frameworks. Its own neurons: token + positional embeddings, a single-head
 * scaled-dot-product attention block (Q/K/V), a tanh hidden layer and a softmax
 * output, with hand-written forward, backprop and Adagrad. Trains on the NyxOS
 * corpus as bytes (V=256) or BPE tokens (-DV=1024, a .tok file). Endgame: run
 * inside NyxOS and generate N / NyxOS code.
 *
 * Model (predicting the token after a B-token window):
 *   x_t = emb(tok_t) + pos_t                          (t = 0..B-1, dim E)
 *   q   = Wq x_{B-1}    k_t = Wk x_t    v_t = Wv x_t   (single query: last pos)
 *   a   = softmax_t( q . k_t / sqrt(E) )              (attention weights)
 *   c   = sum_t a_t v_t                               (context, dim E)
 *   h   = tanh(W1 c + b1)                             (E -> HID)
 *   z   = W2 h + b2 ; softmax ; cross-entropy         (HID -> V)
 *
 * Build:  gcc -O2 -march=native -o oneiros src/oneiros.c -lm        (bytes)
 *         gcc -O2 -march=native -DV=1024 -o oneiros_tok ...          (tokens)
 * Verbs:  train <corpus|.tok> <steps> <out.bin> [resume.bin]
 *         sample <ckpt.bin> <n> [temp] [seed] [vocab.bin=token mode]
 *         gradcheck                                    (numeric vs analytic grad)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifndef V
#define V    256          /* vocab: 256 bytes, or -DV=<n> token ids */
#endif
#ifndef E
#define E    24           /* embedding / model width  (-DE=)        */
#endif
#ifndef B
#define B    8            /* context window in symbols (-DB=)       */
#endif
#ifndef HID
#define HID  256          /* hidden units             (-DHID=)      */
#endif
#define BATCH 32           /* samples averaged per gradient step    */
#define VAL_FRAC 0.10f     /* tail fraction of corpus held out      */

typedef uint16_t sym_t;    /* a symbol: a byte (V=256) or a token id */

/* ---- parameters (the learned weights) ---- */
typedef struct {
    float C[V * E];        /* token embeddings                      */
    float P[B * E];        /* positional embeddings                 */
    float Wq[E * E];       /* query projection                      */
    float Wk[E * E];       /* key projection                        */
    float Wv[E * E];       /* value projection                      */
    float W1[HID * E];     /* attention context -> hidden           */
    float b1[HID];
    float W2[V * HID];     /* hidden -> logits                      */
    float b2[V];
} Params;

typedef struct { Params g2; } Adagrad;   /* Adagrad accumulators */

/* forward activations, kept for the backward pass */
typedef struct {
    float xt[B][E];        /* embed + positional                    */
    float q[E];            /* query (last position)                 */
    float k[B][E];         /* keys                                  */
    float vv[B][E];        /* values                                */
    float score[B];        /* attention scores                      */
    float a[B];            /* attention weights (softmax)           */
    float c[E];            /* context                               */
    float h[HID];          /* hidden                                */
    float probs[V];        /* output distribution                   */
} Act;

/* deterministic RNG so runs are reproducible without external deps */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state >> 32);
}
static inline float frand(void) { return (float)(rnd() / 4294967296.0); }
static float gauss(float sd) {                 /* Box-Muller, for weight init */
    float u1 = frand() + 1e-9f, u2 = frand();
    return sd * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void init_params(Params *p) {
    for (int i = 0; i < V * E; i++) p->C[i] = gauss(0.10f);
    for (int i = 0; i < B * E; i++) p->P[i] = gauss(0.02f);
    float se = 1.0f / sqrtf((float)E);
    for (int i = 0; i < E * E; i++) { p->Wq[i] = gauss(se); p->Wk[i] = gauss(se); p->Wv[i] = gauss(se); }
    for (int i = 0; i < HID * E; i++) p->W1[i] = gauss(se);
    for (int i = 0; i < HID; i++)     p->b1[i] = 0.0f;
    float sh = 1.0f / sqrtf((float)HID);
    for (int i = 0; i < V * HID; i++) p->W2[i] = gauss(sh);
    for (int i = 0; i < V; i++)       p->b2[i] = 0.0f;
}

/* forward pass; fills A, returns cross-entropy loss at `target` */
static float forward(const Params *p, const sym_t *ctx, int target, Act *A) {
    const int L = B - 1;
    const float scale = 1.0f / sqrtf((float)E);
    for (int t = 0; t < B; t++)
        for (int d = 0; d < E; d++)
            A->xt[t][d] = p->C[(int)ctx[t] * E + d] + p->P[t * E + d];

    for (int i = 0; i < E; i++) {
        float s = 0.0f;
        for (int j = 0; j < E; j++) s += p->Wq[i * E + j] * A->xt[L][j];
        A->q[i] = s;
    }
    for (int t = 0; t < B; t++)
        for (int i = 0; i < E; i++) {
            float ki = 0.0f, vi = 0.0f;
            for (int j = 0; j < E; j++) { ki += p->Wk[i * E + j] * A->xt[t][j];
                                          vi += p->Wv[i * E + j] * A->xt[t][j]; }
            A->k[t][i] = ki; A->vv[t][i] = vi;
        }

    float mx = -1e30f;
    for (int t = 0; t < B; t++) {
        float s = 0.0f;
        for (int i = 0; i < E; i++) s += A->q[i] * A->k[t][i];
        s *= scale; A->score[t] = s; if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int t = 0; t < B; t++) { A->a[t] = expf(A->score[t] - mx); sum += A->a[t]; }
    for (int t = 0; t < B; t++) A->a[t] /= sum;

    for (int d = 0; d < E; d++) {
        float cc = 0.0f;
        for (int t = 0; t < B; t++) cc += A->a[t] * A->vv[t][d];
        A->c[d] = cc;
    }
    for (int j = 0; j < HID; j++) {
        float z = p->b1[j];
        for (int d = 0; d < E; d++) z += p->W1[j * E + d] * A->c[d];
        A->h[j] = tanhf(z);
    }
    float zmax = -1e30f;
    for (int k = 0; k < V; k++) {
        float z = p->b2[k];
        for (int j = 0; j < HID; j++) z += p->W2[k * HID + j] * A->h[j];
        A->probs[k] = z; if (z > zmax) zmax = z;
    }
    float zs = 0.0f;
    for (int k = 0; k < V; k++) { A->probs[k] = expf(A->probs[k] - zmax); zs += A->probs[k]; }
    for (int k = 0; k < V; k++) A->probs[k] /= zs;
    return -logf(A->probs[target] + 1e-12f);
}

/* backprop one sample; accumulates into g (caller zeroes between steps) */
static void backward(const Params *p, const sym_t *ctx, int target, const Act *A, Params *g) {
    const int L = B - 1;
    const float scale = 1.0f / sqrtf((float)E);

    float dz[V], dh[HID];
    for (int k = 0; k < V; k++) dz[k] = A->probs[k];
    dz[target] -= 1.0f;
    for (int j = 0; j < HID; j++) dh[j] = 0.0f;
    for (int k = 0; k < V; k++) {
        float d = dz[k]; if (d == 0.0f) continue;
        float *gw = g->W2 + k * HID; const float *w = p->W2 + k * HID;
        for (int j = 0; j < HID; j++) { gw[j] += d * A->h[j]; dh[j] += d * w[j]; }
        g->b2[k] += d;
    }

    float dc[E];
    for (int d = 0; d < E; d++) dc[d] = 0.0f;
    for (int j = 0; j < HID; j++) {
        float draw = dh[j] * (1.0f - A->h[j] * A->h[j]);
        float *gw = g->W1 + j * E; const float *w = p->W1 + j * E;
        for (int d = 0; d < E; d++) { gw[d] += draw * A->c[d]; dc[d] += draw * w[d]; }
        g->b1[j] += draw;
    }

    /* context = sum_t a_t v_t */
    float da[B], dv[B][E];
    for (int t = 0; t < B; t++) {
        float s = 0.0f;
        for (int d = 0; d < E; d++) { s += dc[d] * A->vv[t][d]; dv[t][d] = A->a[t] * dc[d]; }
        da[t] = s;
    }
    /* softmax over scores */
    float sdot = 0.0f;
    for (int t = 0; t < B; t++) sdot += A->a[t] * da[t];
    float dscore[B];
    for (int t = 0; t < B; t++) dscore[t] = A->a[t] * (da[t] - sdot);
    /* score_t = scale * q . k_t */
    float dq[E], dk[B][E];
    for (int i = 0; i < E; i++) dq[i] = 0.0f;
    for (int t = 0; t < B; t++)
        for (int i = 0; i < E; i++) { dq[i] += dscore[t] * A->k[t][i] * scale;
                                      dk[t][i] = dscore[t] * A->q[i] * scale; }

    float dx[B][E];
    for (int t = 0; t < B; t++) for (int d = 0; d < E; d++) dx[t][d] = 0.0f;
    /* q = Wq x_L */
    for (int i = 0; i < E; i++)
        for (int j = 0; j < E; j++) { g->Wq[i * E + j] += dq[i] * A->xt[L][j];
                                      dx[L][j] += p->Wq[i * E + j] * dq[i]; }
    /* k_t = Wk x_t ; v_t = Wv x_t */
    for (int t = 0; t < B; t++)
        for (int i = 0; i < E; i++) {
            float dki = dk[t][i], dvi = dv[t][i];
            for (int j = 0; j < E; j++) {
                g->Wk[i * E + j] += dki * A->xt[t][j]; dx[t][j] += p->Wk[i * E + j] * dki;
                g->Wv[i * E + j] += dvi * A->xt[t][j]; dx[t][j] += p->Wv[i * E + j] * dvi;
            }
        }
    /* x_t = C[tok_t] + P[t] */
    for (int t = 0; t < B; t++)
        for (int d = 0; d < E; d++) { g->C[(int)ctx[t] * E + d] += dx[t][d];
                                      g->P[t * E + d] += dx[t][d]; }
}

/* Adagrad over the whole parameter block as a flat float array */
static void adagrad_step(Params *p, const Params *grad, Adagrad *ad, float lr) {
    float *pf = (float *)p; const float *gf = (const float *)grad; float *af = (float *)&ad->g2;
    size_t n = sizeof(Params) / sizeof(float);
    for (size_t i = 0; i < n; i++) { float gg = gf[i]; af[i] += gg * gg; pf[i] -= lr * gg / (sqrtf(af[i]) + 1e-8f); }
}

/* load a corpus as symbols. A .tok file (magic OTOK1) is uint16 token ids;
 * anything else is raw bytes widened to symbols. */
static sym_t *load_syms(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[5] = {0};
    size_t got = fread(magic, 1, 5, f);
    if (got == 5 && memcmp(magic, "OTOK1", 5) == 0) {
        uint32_t vs; uint64_t nt;
        if (fread(&vs, 4, 1, f) != 1 || fread(&nt, 8, 1, f) != 1) { fprintf(stderr, "bad tok\n"); exit(1); }
        if ((int)vs != V) { fprintf(stderr, "tok vocab %u != build V=%d (recompile -DV=%u)\n", vs, V, vs); exit(1); }
        sym_t *buf = malloc((size_t)nt * sizeof(sym_t));
        size_t n = fread(buf, sizeof(sym_t), (size_t)nt, f);
        fclose(f); *out_n = n;
        fprintf(stderr, "loaded %zu tokens (vocab %u)\n", n, vs);
        return buf;
    }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *raw = malloc((size_t)sz);
    size_t n = fread(raw, 1, (size_t)sz, f);
    fclose(f);
    sym_t *buf = malloc(n * sizeof(sym_t));
    for (size_t i = 0; i < n; i++) buf[i] = raw[i];
    free(raw); *out_n = n;
    fprintf(stderr, "loaded %zu bytes\n", n);
    return buf;
}

static void save(const char *path, const Params *p) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int32_t dims[4] = { V, E, B, HID };
    fwrite("ONEIROS2", 1, 8, f);
    fwrite(dims, sizeof(dims), 1, f);
    fwrite(p, sizeof(Params), 1, f);
    fclose(f);
}

static int load(const char *path, Params *p) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[8]; int32_t dims[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "ONEIROS2", 8) != 0) { fclose(f); return 0; }
    if (fread(dims, sizeof(dims), 1, f) != 1) { fclose(f); return 0; }
    if (dims[0] != V || dims[1] != E || dims[2] != B || dims[3] != HID) {
        fprintf(stderr, "checkpoint dims mismatch, starting fresh\n"); fclose(f); return 0;
    }
    int ok = fread(p, sizeof(Params), 1, f) == 1;
    fclose(f);
    return ok;
}

/* average cross-entropy over `count` random windows in [lo, hi); leaves the
 * training RNG stream untouched so runs stay reproducible */
static double eval_loss(const Params *p, const sym_t *text, size_t lo, size_t hi, int count) {
    if (hi < lo + B + 1) return 0.0;
    size_t span = hi - lo - B - 1; if (span == 0) span = 1;
    uint64_t saved = rng_state;
    double sum = 0.0;
    Act A;
    for (int i = 0; i < count; i++) {
        size_t pos = lo + (size_t)(((uint64_t)rnd() << 20 ^ rnd()) % span);
        sum += forward(p, text + pos, (int)text[pos + B], &A);
    }
    rng_state = saved;
    return sum / count;
}

static void cmd_train(int argc, char **argv) {
    const char *corpus = argv[2];
    long steps         = atol(argv[3]);
    const char *out    = argv[4];
    const char *in     = (argc > 5) ? argv[5] : NULL;

    size_t n;
    sym_t *text = load_syms(corpus, &n);
    if (n < (size_t)(B + 2)) { fprintf(stderr, "corpus too small\n"); exit(1); }

    Params *p    = malloc(sizeof(Params));
    Params *grad = malloc(sizeof(Params));
    Adagrad *ad  = calloc(1, sizeof(Adagrad));
    if (in && load(in, p)) fprintf(stderr, "resumed from %s\n", in);
    else                   { init_params(p); fprintf(stderr, "fresh init\n"); }

    size_t train_end = (size_t)(n * (1.0 - (double)VAL_FRAC));
    if (train_end < (size_t)(B + 2)) train_end = n;
    size_t train_span = train_end - B - 1;
    fprintf(stderr, "corpus %zu symbols: train [0,%zu) val [%zu,%zu)  params %zu\n",
            n, train_end, train_end, n, sizeof(Params) / sizeof(float));

    double run = 0.0; long runc = 0;
    float lr = 0.10f;
    clock_t t0 = clock();
    long batches = steps / BATCH;
    const float invb = 1.0f / BATCH;
    size_t np = sizeof(Params) / sizeof(float);
    Act A;
    for (long bi = 0; bi < batches; bi++) {
        memset(grad, 0, sizeof(Params));
        double bloss = 0.0;
        for (int j = 0; j < BATCH; j++) {
            size_t pos = (size_t)(((uint64_t)rnd() << 20 ^ rnd()) % train_span);
            const sym_t *ctx = text + pos;
            int target = text[pos + B];
            bloss += forward(p, ctx, target, &A);
            backward(p, ctx, target, &A, grad);
        }
        float *gf = (float *)grad;
        for (size_t i = 0; i < np; i++) gf[i] *= invb;
        adagrad_step(p, grad, ad, lr);

        run += bloss * invb; runc++;
        long done = (bi + 1) * (long)BATCH;
        if (done % 20000 < BATCH) {
            double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
            double vl = eval_loss(p, text, train_end, n, 2000);
            fprintf(stderr, "step %8ld  train %.4f  val %.4f  (%.0f samp/s)\n",
                    done, run / runc, vl, done / (sec + 1e-9));
            run = 0.0; runc = 0;
        }
    }
    save(out, p);
    double vl = eval_loss(p, text, train_end, n, 20000);
    fprintf(stderr, "saved %s   final val loss (20k windows): %.4f\n", out, vl);
    free(text); free(p); free(grad); free(ad);
}

static int sample_dist(const float *probs) {
    float r = frand(), c = 0.0f;
    for (int k = 0; k < V; k++) { c += probs[k]; if (r <= c) return k; }
    return V - 1;
}

/* ---- BPE tokenizer (vocab.bin) for token-mode sampling ---- */
typedef struct {
    int vocab_size, num_merges;
    uint16_t *ma, *mb;
    uint8_t **exp; uint16_t *el;
} Tok;

static int tok_load(const char *path, Tok *t) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[5];
    if (fread(magic, 1, 5, f) != 5 || memcmp(magic, "OBPE1", 5)) { fclose(f); return 0; }
    uint32_t vs, nm;
    if (fread(&vs, 4, 1, f) != 1 || fread(&nm, 4, 1, f) != 1) { fclose(f); return 0; }
    t->vocab_size = vs; t->num_merges = nm;
    t->ma = malloc(nm * sizeof(uint16_t)); t->mb = malloc(nm * sizeof(uint16_t));
    for (uint32_t k = 0; k < nm; k++) if (fread(&t->ma[k],2,1,f)!=1 || fread(&t->mb[k],2,1,f)!=1) {fclose(f);return 0;}
    t->exp = malloc(vs * sizeof(uint8_t *)); t->el = malloc(vs * sizeof(uint16_t));
    for (uint32_t i = 0; i < vs; i++) {
        uint16_t l; if (fread(&l,2,1,f)!=1) {fclose(f);return 0;}
        t->el[i] = l; t->exp[i] = malloc(l ? l : 1);
        if (l && fread(t->exp[i],1,l,f)!=l) {fclose(f);return 0;}
    }
    fclose(f);
    return 1;
}

static size_t tok_encode(const Tok *t, const char *s, size_t slen, sym_t *out, size_t cap) {
    size_t len = 0;
    for (size_t i = 0; i < slen && len < cap; i++) out[len++] = (unsigned char)s[i];
    for (int k = 0; k < t->num_merges; k++) {
        uint16_t a = t->ma[k], b = t->mb[k], c = (uint16_t)(256 + k);
        size_t j = 0;
        for (size_t i = 0; i < len; i++) {
            if (i + 1 < len && out[i] == a && out[i + 1] == b) { out[j++] = c; i++; }
            else out[j++] = out[i];
        }
        len = j;
    }
    return len;
}

static void cmd_sample(int argc, char **argv) {
    const char *ckpt = argv[2];
    int howmany      = atoi(argv[3]);
    float temp       = (argc > 4) ? (float)atof(argv[4]) : 0.8f;
    const char *seed = (argc > 5) ? argv[5] : "void ";
    const char *vpath= (argc > 6) ? argv[6] : NULL;

    Params *p = malloc(sizeof(Params));
    if (!load(ckpt, p)) { fprintf(stderr, "cannot load %s\n", ckpt); exit(1); }

    Tok tok; int token_mode = 0;
    if (vpath) {
        if (!tok_load(vpath, &tok)) { fprintf(stderr, "cannot load vocab %s\n", vpath); exit(1); }
        if (tok.vocab_size != V) { fprintf(stderr, "vocab %d != build V=%d\n", tok.vocab_size, V); exit(1); }
        token_mode = 1;
    }

    sym_t ctx[B];
    if (token_mode) {
        sym_t enc[8192];
        size_t m = tok_encode(&tok, seed, strlen(seed), enc, 8192);
        for (int t = 0; t < B; t++) { long idx = (long)m - B + t; ctx[t] = (idx >= 0) ? enc[idx] : (sym_t)' '; }
    } else {
        int sl = (int)strlen(seed);
        for (int t = 0; t < B; t++) { int idx = sl - B + t; ctx[t] = (idx >= 0) ? (sym_t)(unsigned char)seed[idx] : (sym_t)' '; }
    }
    fputs(seed, stdout);

    Act A;
    rng_state ^= (uint64_t)time(NULL) * 0x2545F4914F6CDD1DULL;
    for (int i = 0; i < howmany; i++) {
        forward(p, ctx, 0, &A);
        float maxlp = -1e30f;
        for (int k = 0; k < V; k++) { float lp = logf(A.probs[k] + 1e-12f) / temp;
                                      A.probs[k] = lp; if (lp > maxlp) maxlp = lp; }
        float sum = 0.0f;
        for (int k = 0; k < V; k++) { A.probs[k] = expf(A.probs[k] - maxlp); sum += A.probs[k]; }
        for (int k = 0; k < V; k++) A.probs[k] /= sum;
        int nx = sample_dist(A.probs);
        if (token_mode) fwrite(tok.exp[nx], 1, tok.el[nx], stdout);
        else            putchar(nx);
        memmove(ctx, ctx + 1, (B - 1) * sizeof(sym_t));
        ctx[B - 1] = (sym_t)nx;
    }
    putchar('\n');
    free(p);
}

/* numerically verify backprop: central-difference grad vs analytic grad.
 * Warms up first (a uniform fresh softmax gives ~zero gradient to almost every
 * W2 entry, whose finite difference is then pure float rounding noise), and only
 * scores parameters whose analytic gradient is non-negligible. */
static void cmd_gradcheck(void) {
    Params *p = malloc(sizeof(Params));
    Params *g = malloc(sizeof(Params));
    Adagrad *ad = calloc(1, sizeof(Adagrad));
    init_params(p);
    Act A, tmp;
    sym_t ctx[B];

    for (int s = 0; s < 500; s++) {                 /* warm up on random samples */
        for (int t = 0; t < B; t++) ctx[t] = (sym_t)(rnd() % V);
        int tg = (int)(rnd() % V);
        memset(g, 0, sizeof(Params));
        forward(p, ctx, tg, &A); backward(p, ctx, tg, &A, g);
        adagrad_step(p, g, ad, 0.05f);
    }

    for (int t = 0; t < B; t++) ctx[t] = (sym_t)(rnd() % V);
    int target = (int)(rnd() % V);
    memset(g, 0, sizeof(Params));
    forward(p, ctx, target, &A);
    backward(p, ctx, target, &A, g);

    float *pf = (float *)p; const float *gf = (const float *)g;
    size_t np = sizeof(Params) / sizeof(float);
    const float eps = 2e-3f;
    int informative = 0, good = 0; double meanrel = 0.0, maxrel = 0.0;
    for (int n = 0; n < 200000 && informative < 4000; n++) {
        size_t i = (size_t)(((uint64_t)rnd() << 20 ^ rnd()) % np);
        double ana = gf[i];
        if (fabs(ana) < 1e-3) continue;             /* skip float-noise params  */
        float save = pf[i];
        pf[i] = save + eps; float lp = forward(p, ctx, target, &tmp);
        pf[i] = save - eps; float lm = forward(p, ctx, target, &tmp);
        pf[i] = save;
        double num = (lp - lm) / (2.0 * eps);
        double rel = fabs(num - ana) / (fabs(num) + fabs(ana) + 1e-9);
        meanrel += rel; if (rel > maxrel) maxrel = rel; if (rel < 0.05) good++; informative++;
    }
    printf("gradcheck: %d informative params  mean rel %.5f  max %.5f  within 5%%: %.1f%%\n",
           informative, meanrel / informative, maxrel, 100.0 * good / informative);
    printf("%s\n", (meanrel / informative < 0.02) ? "BACKPROP OK" : "BACKPROP SUSPECT");
    free(p); free(g); free(ad);
}

int main(int argc, char **argv) {
    if (argc >= 5 && strcmp(argv[1], "train") == 0)  { cmd_train(argc, argv);  return 0; }
    if (argc >= 4 && strcmp(argv[1], "sample") == 0) { cmd_sample(argc, argv); return 0; }
    if (argc >= 2 && strcmp(argv[1], "gradcheck") == 0) { cmd_gradcheck(); return 0; }
    fprintf(stderr,
        "usage:\n"
        "  %s train  <corpus|corpus.tok> <steps> <out.bin> [resume.bin]\n"
        "  %s sample <ckpt.bin> <n> [temp] [seed] [vocab.bin=token mode]\n"
        "  %s gradcheck\n", argv[0], argv[0], argv[0]);
    return 1;
}
