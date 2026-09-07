/*
 * Oneiros - an independent generative neural network for NyxOS.
 *
 * Iteration 1: a from-scratch character-level neural language model.
 * No ML frameworks. The "neurons" are here in the open: an embedding
 * table, one tanh hidden layer, a softmax output, and hand-written
 * forward / backprop / Adagrad training. It learns to continue NyxOS's
 * own C source, one byte at a time. Later iterations grow this toward a
 * transformer and fixed-point inference that runs inside NyxOS, whose
 * purpose is to generate N / NyxOS code from within the OS itself.
 *
 * Architecture (Bengio-style neural LM):
 *   context of B previous bytes
 *     -> embedding lookup (V x E), concatenated to IN = B*E
 *     -> hidden  h = tanh(W1 . x + b1)          [IN -> HID]
 *     -> logits  z = W2 . h + b2                 [HID -> V]
 *     -> softmax -> cross-entropy against the true next byte
 *
 * Build:  gcc -O2 -o oneiros src/oneiros.c -lm
 * Train:  ./oneiros train data/corpus.txt 300000 data/oneiros.bin [in.bin]
 * Sample: ./oneiros sample data/oneiros.bin 400 0.8 "void "
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define V    256          /* vocabulary: raw bytes                 */
#define E    24           /* embedding dimension                   */
#define B    8            /* context window (bytes of history)     */
#define IN   (B * E)      /* concatenated context vector           */
#define HID  256          /* hidden units                          */

/* ---- parameters (the learned weights) ---- */
typedef struct {
    float C[V * E];       /* byte embeddings                       */
    float W1[HID * IN];   /* input  -> hidden                      */
    float b1[HID];
    float W2[V * HID];    /* hidden -> logits                      */
    float b2[V];
} Params;

/* ---- Adagrad accumulators (one per parameter) ---- */
typedef struct { Params g2; } Adagrad;

/* deterministic RNG so runs are reproducible without external deps */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state >> 32);
}
static inline float frand(void) { return (float)(rnd() / 4294967296.0); }
/* gaussian via Box-Muller, for weight init */
static float gauss(float sd) {
    float u1 = frand() + 1e-9f, u2 = frand();
    return sd * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void init_params(Params *p) {
    for (int i = 0; i < V * E; i++)   p->C[i]  = gauss(0.10f);
    /* Xavier-ish scaling for the linear layers */
    float s1 = 1.0f / sqrtf((float)IN);
    float s2 = 1.0f / sqrtf((float)HID);
    for (int i = 0; i < HID * IN; i++) p->W1[i] = gauss(s1);
    for (int i = 0; i < HID; i++)      p->b1[i] = 0.0f;
    for (int i = 0; i < V * HID; i++)  p->W2[i] = gauss(s2);
    for (int i = 0; i < V; i++)        p->b2[i] = 0.0f;
}

/* forward pass; fills x (input), h (hidden), probs (softmax). returns loss. */
static float forward(const Params *p, const unsigned char *ctx, int target,
                     float *x, float *h, float *probs) {
    for (int t = 0; t < B; t++) {
        const float *emb = p->C + (int)ctx[t] * E;
        memcpy(x + t * E, emb, E * sizeof(float));
    }
    for (int j = 0; j < HID; j++) {
        const float *w = p->W1 + j * IN;
        float a = p->b1[j];
        for (int i = 0; i < IN; i++) a += w[i] * x[i];
        h[j] = tanhf(a);
    }
    float maxz = -1e30f;
    for (int k = 0; k < V; k++) {
        const float *w = p->W2 + k * HID;
        float z = p->b2[k];
        for (int j = 0; j < HID; j++) z += w[j] * h[j];
        probs[k] = z;
        if (z > maxz) maxz = z;
    }
    float sum = 0.0f;
    for (int k = 0; k < V; k++) { probs[k] = expf(probs[k] - maxz); sum += probs[k]; }
    float inv = 1.0f / sum;
    for (int k = 0; k < V; k++) probs[k] *= inv;
    return -logf(probs[target] + 1e-12f);
}

/* backprop one sample into grad; accumulates (caller zeroes between steps) */
static void backward(const Params *p, const unsigned char *ctx, int target,
                     const float *x, const float *h, const float *probs,
                     Params *grad) {
    float dz[V], dh[HID];
    for (int k = 0; k < V; k++) dz[k] = probs[k];
    dz[target] -= 1.0f;

    for (int j = 0; j < HID; j++) dh[j] = 0.0f;
    for (int k = 0; k < V; k++) {
        float d = dz[k];
        if (d == 0.0f) continue;
        float *gw = grad->W2 + k * HID;
        const float *w = p->W2 + k * HID;
        for (int j = 0; j < HID; j++) { gw[j] += d * h[j]; dh[j] += d * w[j]; }
        grad->b2[k] += d;
    }

    float dx[IN];
    for (int i = 0; i < IN; i++) dx[i] = 0.0f;
    for (int j = 0; j < HID; j++) {
        float draw = dh[j] * (1.0f - h[j] * h[j]);   /* tanh' */
        float *gw = grad->W1 + j * IN;
        const float *w = p->W1 + j * IN;
        for (int i = 0; i < IN; i++) { gw[i] += draw * x[i]; dx[i] += draw * w[i]; }
        grad->b1[j] += draw;
    }
    for (int t = 0; t < B; t++) {
        float *ge = grad->C + (int)ctx[t] * E;
        const float *dxt = dx + t * E;
        for (int i = 0; i < E; i++) ge[i] += dxt[i];
    }
}

/* Adagrad update over the whole parameter block treated as a flat float array */
static void adagrad_step(Params *p, const Params *grad, Adagrad *ad, float lr) {
    float       *pf = (float *)p;
    const float *gf = (const float *)grad;
    float       *af = (float *)&ad->g2;
    size_t n = sizeof(Params) / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        float g = gf[i];
        af[i] += g * g;
        pf[i] -= lr * g / (sqrtf(af[i]) + 1e-8f);
    }
}

static char *read_corpus(const char *path, size_t cap, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char *buf = malloc(cap);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    *out_n = n;
    return buf;
}

static void save(const char *path, const Params *p) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    const char magic[8] = "ONEIROS1";
    int32_t dims[4] = { V, E, B, HID };
    fwrite(magic, 1, 8, f);
    fwrite(dims, sizeof(dims), 1, f);
    fwrite(p, sizeof(Params), 1, f);
    fclose(f);
}

static int load(const char *path, Params *p) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[8]; int32_t dims[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "ONEIROS1", 8) != 0) { fclose(f); return 0; }
    if (fread(dims, sizeof(dims), 1, f) != 1) { fclose(f); return 0; }
    if (dims[0] != V || dims[1] != E || dims[2] != B || dims[3] != HID) {
        fprintf(stderr, "checkpoint dims mismatch, starting fresh\n"); fclose(f); return 0;
    }
    int ok = fread(p, sizeof(Params), 1, f) == 1;
    fclose(f);
    return ok;
}

static void cmd_train(int argc, char **argv) {
    const char *corpus = argv[2];
    long steps         = atol(argv[3]);
    const char *out    = argv[4];
    const char *in     = (argc > 5) ? argv[5] : NULL;

    size_t n;
    char *text = read_corpus(corpus, 64u * 1024 * 1024, &n);   /* whole corpus (<=64 MB) */
    if (n < (size_t)(B + 2)) { fprintf(stderr, "corpus too small\n"); exit(1); }

    Params *p    = malloc(sizeof(Params));
    Params *grad = malloc(sizeof(Params));
    Adagrad *ad  = calloc(1, sizeof(Adagrad));
    if (in && load(in, p)) fprintf(stderr, "resumed from %s\n", in);
    else                   { init_params(p); fprintf(stderr, "fresh init\n"); }

    float x[IN], h[HID], probs[V];
    double run = 0.0; long runc = 0;
    float lr = 0.10f;
    clock_t t0 = clock();

    for (long s = 0; s < steps; s++) {
        size_t pos = (size_t)(((uint64_t)rnd() << 20 ^ rnd()) % (n - B - 1));
        const unsigned char *ctx = (const unsigned char *)text + pos;
        int target = (unsigned char)text[pos + B];

        float loss = forward(p, ctx, target, x, h, probs);
        memset(grad, 0, sizeof(Params));
        backward(p, ctx, target, x, h, probs, grad);
        adagrad_step(p, grad, ad, lr);

        run += loss; runc++;
        if ((s + 1) % 20000 == 0) {
            double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
            fprintf(stderr, "step %8ld  loss %.4f  (%.0f samp/s)\n",
                    s + 1, run / runc, (s + 1) / (sec + 1e-9));
            run = 0.0; runc = 0;
        }
    }
    save(out, p);
    fprintf(stderr, "saved %s\n", out);
    free(text); free(p); free(grad); free(ad);
}

static int sample_dist(const float *probs, float temp) {
    /* temperature already folded in by caller via re-softmax; plain sample here */
    float r = frand(), c = 0.0f;
    for (int k = 0; k < V; k++) { c += probs[k]; if (r <= c) return k; }
    return V - 1;
    (void)temp;
}

static void cmd_sample(int argc, char **argv) {
    const char *ckpt = argv[2];
    int howmany      = atoi(argv[3]);
    float temp       = (argc > 4) ? (float)atof(argv[4]) : 0.8f;
    const char *seed = (argc > 5) ? argv[5] : "void ";

    Params *p = malloc(sizeof(Params));
    if (!load(ckpt, p)) { fprintf(stderr, "cannot load %s\n", ckpt); exit(1); }

    unsigned char ctx[B];
    int sl = (int)strlen(seed);
    for (int t = 0; t < B; t++) {
        int idx = sl - B + t;
        ctx[t] = (idx >= 0) ? (unsigned char)seed[idx] : (unsigned char)' ';
    }
    fputs(seed, stdout);

    float x[IN], h[HID], probs[V];
    rng_state ^= (uint64_t)time(NULL) * 0x2545F4914F6CDD1DULL;
    for (int i = 0; i < howmany; i++) {
        forward(p, ctx, 0, x, h, probs);           /* target unused for logits */
        /* re-apply temperature: probs are p=softmax(z); recover via log then re-softmax */
        float maxlp = -1e30f;
        for (int k = 0; k < V; k++) { float lp = logf(probs[k] + 1e-12f) / temp;
                                      probs[k] = lp; if (lp > maxlp) maxlp = lp; }
        float sum = 0.0f;
        for (int k = 0; k < V; k++) { probs[k] = expf(probs[k] - maxlp); sum += probs[k]; }
        for (int k = 0; k < V; k++) probs[k] /= sum;

        int nx = sample_dist(probs, temp);
        putchar(nx);
        memmove(ctx, ctx + 1, B - 1);
        ctx[B - 1] = (unsigned char)nx;
    }
    putchar('\n');
    free(p);
}

int main(int argc, char **argv) {
    if (argc >= 5 && strcmp(argv[1], "train") == 0)  { cmd_train(argc, argv);  return 0; }
    if (argc >= 4 && strcmp(argv[1], "sample") == 0) { cmd_sample(argc, argv); return 0; }
    fprintf(stderr,
        "usage:\n"
        "  %s train  <corpus> <steps> <out.bin> [resume.bin]\n"
        "  %s sample <ckpt.bin> <nchars> [temperature] [seedstr]\n", argv[0], argv[0]);
    return 1;
}
