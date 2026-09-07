/*
 * bpe.c - a from-scratch byte-pair-encoding tokeniser for Oneiros.
 *
 * No libraries. Learns merges from the NyxOS code corpus so the model can
 * reason in C tokens (identifiers, keywords, "int ", "return", "0x", "();")
 * instead of raw bytes. Byte-level BPE (GPT-2 style): start from 256 byte
 * tokens, repeatedly merge the most frequent adjacent pair.
 *
 *   bpe learn  <corpus> <num_merges> <vocab.bin> <corpus.tok>
 *   bpe decode <vocab.bin> <corpus.tok>          # -> stdout (round-trip check)
 *   bpe stats  <vocab.bin>                        # print the learned vocab
 *
 * vocab.bin: "OBPE1" | u32 vocab_size | u32 num_merges |
 *            num_merges x (u16 a, u16 b)          # merges in rank order, c=256+k
 *            vocab_size x (u16 len, len bytes)     # each token's expansion
 * corpus.tok: "OTOK1" | u32 vocab_size | u64 ntokens | ntokens x u16 id
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint8_t *slurp(const char *path, size_t cap, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint8_t *buf = malloc(cap ? cap : 1);
    size_t n = cap ? fread(buf, 1, cap, f) : 0;
    fclose(f);
    *out_n = n;
    return buf;
}

/* greedy left-to-right rewrite: every adjacent (a,b) becomes c. returns new len */
static size_t merge_pass(uint16_t *seq, size_t len, uint16_t a, uint16_t b, uint16_t c) {
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (i + 1 < len && seq[i] == a && seq[i + 1] == b) { seq[j++] = c; i++; }
        else                                                 seq[j++] = seq[i];
    }
    return j;
}

static void cmd_learn(char **argv) {
    const char *corpus = argv[2];
    int num_merges     = atoi(argv[3]);
    const char *vpath  = argv[4];
    const char *tpath  = argv[5];
    int vocab_size     = 256 + num_merges;
    int VCAP           = vocab_size;              /* pair index base */

    /* learn merges on a subset (merges generalise); encode the FULL corpus after */
    size_t subn;
    uint8_t *sub = slurp(corpus, 4u * 1024 * 1024, &subn);
    uint16_t *seq = malloc(subn * sizeof(uint16_t));
    for (size_t i = 0; i < subn; i++) seq[i] = sub[i];
    size_t len = subn;
    free(sub);

    uint16_t *ma = malloc(num_merges * sizeof(uint16_t));   /* merge left  */
    uint16_t *mb = malloc(num_merges * sizeof(uint16_t));   /* merge right */
    int *counts = malloc((size_t)VCAP * VCAP * sizeof(int));
    int done = 0;
    for (int k = 0; k < num_merges; k++) {
        memset(counts, 0, (size_t)VCAP * VCAP * sizeof(int));
        for (size_t i = 0; i + 1 < len; i++)
            counts[(size_t)seq[i] * VCAP + seq[i + 1]]++;
        int best = -1; long bestc = 1;             /* need freq >= 2 to bother */
        for (size_t idx = 0; idx < (size_t)VCAP * VCAP; idx++)
            if (counts[idx] > bestc) { bestc = counts[idx]; best = (int)idx; }
        if (best < 0) { fprintf(stderr, "no pair >=2 at merge %d; stopping\n", k); break; }
        uint16_t a = (uint16_t)(best / VCAP), b = (uint16_t)(best % VCAP);
        uint16_t c = (uint16_t)(256 + k);
        ma[k] = a; mb[k] = b;
        len = merge_pass(seq, len, a, b, c);
        done = k + 1;
        if (k < 8 || (k + 1) % 128 == 0)
            fprintf(stderr, "merge %4d: (%u,%u)->%u  freq %ld  seqlen %zu\n",
                    k, a, b, c, bestc, len);
    }
    free(counts); free(seq);
    num_merges = done; vocab_size = 256 + done;

    /* build each token's byte expansion (for decode) */
    uint8_t **exp = malloc(vocab_size * sizeof(uint8_t *));
    size_t   *el  = malloc(vocab_size * sizeof(size_t));
    for (int i = 0; i < 256; i++) { exp[i] = malloc(1); exp[i][0] = (uint8_t)i; el[i] = 1; }
    for (int k = 0; k < num_merges; k++) {
        int c = 256 + k; uint16_t a = ma[k], b = mb[k];
        el[c] = el[a] + el[b];
        exp[c] = malloc(el[c]);
        memcpy(exp[c],          exp[a], el[a]);
        memcpy(exp[c] + el[a],  exp[b], el[b]);
    }

    /* encode the FULL corpus by applying merges in rank order */
    size_t fulln;
    uint8_t *full = slurp(corpus, 64u * 1024 * 1024, &fulln);
    uint16_t *fseq = malloc(fulln * sizeof(uint16_t));
    for (size_t i = 0; i < fulln; i++) fseq[i] = full[i];
    free(full);
    size_t flen = fulln;
    for (int k = 0; k < num_merges; k++)
        flen = merge_pass(fseq, flen, ma[k], mb[k], (uint16_t)(256 + k));

    /* write vocab.bin */
    FILE *vf = fopen(vpath, "wb");
    if (!vf) { perror(vpath); exit(1); }
    uint32_t vs = vocab_size, nm = num_merges;
    fwrite("OBPE1", 1, 5, vf);
    fwrite(&vs, 4, 1, vf); fwrite(&nm, 4, 1, vf);
    for (int k = 0; k < num_merges; k++) { fwrite(&ma[k], 2, 1, vf); fwrite(&mb[k], 2, 1, vf); }
    for (int i = 0; i < vocab_size; i++) {
        uint16_t l = (uint16_t)el[i];
        fwrite(&l, 2, 1, vf); fwrite(exp[i], 1, l, vf);
    }
    fclose(vf);

    /* write corpus.tok */
    FILE *tf = fopen(tpath, "wb");
    if (!tf) { perror(tpath); exit(1); }
    uint64_t nt = flen;
    fwrite("OTOK1", 1, 5, tf);
    fwrite(&vs, 4, 1, tf); fwrite(&nt, 8, 1, tf);
    fwrite(fseq, 2, flen, tf);
    fclose(tf);

    fprintf(stderr,
        "learned %d merges, vocab %d. corpus %zu bytes -> %zu tokens "
        "(%.3f bytes/token). wrote %s + %s\n",
        num_merges, vocab_size, fulln, flen, (double)fulln / (double)flen, vpath, tpath);
}

/* load vocab expansions for decode; returns vocab_size, fills exp/el (callee frees) */
static int load_vocab(const char *path, uint8_t ***exp_out, size_t **el_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[5]; uint32_t vs, nm;
    if (fread(magic, 1, 5, f) != 5 || memcmp(magic, "OBPE1", 5)) { fprintf(stderr, "bad vocab\n"); exit(1); }
    fread(&vs, 4, 1, f); fread(&nm, 4, 1, f);
    fseek(f, (long)nm * 4, SEEK_CUR);                 /* skip merges */
    uint8_t **exp = malloc(vs * sizeof(uint8_t *));
    size_t   *el  = malloc(vs * sizeof(size_t));
    for (uint32_t i = 0; i < vs; i++) {
        uint16_t l; fread(&l, 2, 1, f);
        exp[i] = malloc(l ? l : 1); el[i] = l;
        fread(exp[i], 1, l, f);
    }
    fclose(f);
    *exp_out = exp; *el_out = el;
    return (int)vs;
}

static void cmd_decode(char **argv) {
    uint8_t **exp; size_t *el;
    load_vocab(argv[2], &exp, &el);
    FILE *tf = fopen(argv[3], "rb");
    if (!tf) { perror(argv[3]); exit(1); }
    char magic[5]; uint32_t vs; uint64_t nt;
    if (fread(magic, 1, 5, tf) != 5 || memcmp(magic, "OTOK1", 5)) { fprintf(stderr, "bad tok\n"); exit(1); }
    fread(&vs, 4, 1, tf); fread(&nt, 8, 1, tf);
    for (uint64_t i = 0; i < nt; i++) {
        uint16_t id; if (fread(&id, 2, 1, tf) != 1) break;
        fwrite(exp[id], 1, el[id], stdout);
    }
    fclose(tf);
}

static void cmd_stats(char **argv) {
    uint8_t **exp; size_t *el;
    int vs = load_vocab(argv[2], &exp, &el);
    fprintf(stderr, "vocab %d. sample of learned tokens (>=2 bytes):\n", vs);
    int shown = 0;
    for (int i = 256; i < vs && shown < 60; i++) {
        if (el[i] < 2) continue;
        fputs("  [", stderr);
        for (size_t j = 0; j < el[i]; j++) {
            uint8_t c = exp[i][j];
            if (c == '\n') fputs("\\n", stderr);
            else if (c == '\t') fputs("\\t", stderr);
            else fputc(c, stderr);
        }
        fputs("]\n", stderr); shown++;
    }
}

int main(int argc, char **argv) {
    if (argc == 6 && !strcmp(argv[1], "learn"))  { cmd_learn(argv);  return 0; }
    if (argc == 4 && !strcmp(argv[1], "decode")) { cmd_decode(argv); return 0; }
    if (argc == 3 && !strcmp(argv[1], "stats"))  { cmd_stats(argv);  return 0; }
    fprintf(stderr, "usage:\n  bpe learn <corpus> <num_merges> <vocab.bin> <corpus.tok>\n"
                    "  bpe decode <vocab.bin> <corpus.tok>\n  bpe stats <vocab.bin>\n");
    return 1;
}
