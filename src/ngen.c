/*
 * ngen.c - Oneiros inference, self-contained for running INSIDE NyxOS.
 *
 * Loads the champion MLP checkpoint (ONEIROS1) + BPE vocab and generates
 * NyxOS-flavoured code. Uses ONLY malloc / fopen / fread / fwrite (which NyxOS
 * userland provides) plus its OWN expf / logf / tanhf — NyxOS's libm has
 * sinf/cosf/sqrtf but NOT exp/log/tanh, and the kernel is -mno-sse, so Oneiros
 * brings its own float math. No libm: build links without -lm.
 *
 * Host build/verify:  gcc -O2 -DV=2048 -o ngen src/ngen.c        (note: no -lm)
 * Run:  ./ngen data/oneiros_2048.bin data/vocab2048.bin 80 0.6 "static void "
 *
 * This is the in-OS-ready inference (I6). Actually dropping it into NyxOS is a
 * public step and needs the maintainer's OK first.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifndef V
#define V   2048
#endif
#ifndef E
#define E   24
#endif
#ifndef B
#define B   8
#endif
#define IN  (B * E)
#ifndef HID
#define HID 256
#endif

typedef uint16_t sym_t;

typedef struct {
    float C[V * E];
    float W1[HID * IN];
    float b1[HID];
    float W2[V * HID];
    float b2[V];
} Params;

/* ---- own float math (no libm; NyxOS userland lacks exp/log/tanh) ---- */
static float k_expf(float x) {
    if (x > 88.0f)  return 3.0e38f;
    if (x < -88.0f) return 0.0f;
    const float LOG2E = 1.44269504f, LN2 = 0.6931471805f;
    int n = (int)(x * LOG2E + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)n * LN2;                 /* r in ~[-0.35, 0.35] */
    float p = 1.0f + r*(1.0f + r*(0.5f + r*(0.16666667f + r*(0.041666668f + r*0.008333334f))));
    union { float f; int32_t i; } u;
    u.i = (int32_t)((n + 127) << 23);             /* 2^n */
    return p * u.f;
}
static float k_logf(float x) {
    if (x <= 0.0f) return -88.0f;
    union { float f; int32_t i; } u; u.f = x;
    int e = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x807FFFFF) | 0x3F800000;        /* mantissa in [1,2) */
    float m = u.f, t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
    float s = t * (2.0f + t2*(0.6666667f + t2*(0.4f + t2*0.2857143f)));
    return (float)e * 0.6931471805f + s;
}
static float k_tanhf(float x) {
    if (x > 15.0f)  return 1.0f;
    if (x < -15.0f) return -1.0f;
    float e = k_expf(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

/* deterministic RNG */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (uint32_t)(rng>>32); }
static float frand(void) { return (float)(rnd() / 4294967296.0); }

/* MLP forward: concat embeddings -> tanh hidden -> softmax (matches oneiros.c) */
static void forward(const Params *p, const sym_t *ctx, float *probs) {
    float x[IN], h[HID];
    for (int t = 0; t < B; t++)
        memcpy(x + t*E, p->C + (int)ctx[t]*E, E*sizeof(float));
    for (int j = 0; j < HID; j++) {
        const float *w = p->W1 + j*IN; float a = p->b1[j];
        for (int i = 0; i < IN; i++) a += w[i]*x[i];
        h[j] = k_tanhf(a);
    }
    float mx = -1e30f;
    for (int k = 0; k < V; k++) {
        const float *w = p->W2 + k*HID; float z = p->b2[k];
        for (int j = 0; j < HID; j++) z += w[j]*h[j];
        probs[k] = z; if (z > mx) mx = z;
    }
    float s = 0.0f;
    for (int k = 0; k < V; k++) { probs[k] = k_expf(probs[k]-mx); s += probs[k]; }
    for (int k = 0; k < V; k++) probs[k] /= s;
}

static int load_ckpt(const char *path, Params *p) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char magic[8]; int32_t dims[4];
    if (fread(magic,1,8,f)!=8 || memcmp(magic,"ONEIROS1",8)) { fclose(f); return 0; }
    if (fread(dims,sizeof(dims),1,f)!=1) { fclose(f); return 0; }
    if (dims[0]!=V || dims[1]!=E || dims[2]!=B || dims[3]!=HID) {
        fprintf(stderr,"ckpt dims %d/%d/%d/%d != build %d/%d/%d/%d\n",
                dims[0],dims[1],dims[2],dims[3],V,E,B,HID); fclose(f); return 0;
    }
    int ok = fread(p,sizeof(Params),1,f)==1; fclose(f); return ok;
}

/* BPE vocab: merges (for seed encode) + expansions (for decode) */
typedef struct { int vs, nm; uint16_t *ma,*mb; uint8_t **ex; uint16_t *el; } Tok;
static int tok_load(const char *path, Tok *t) {
    FILE *f = fopen(path,"rb"); if (!f) return 0;
    char m[5]; uint32_t vs,nm;
    if (fread(m,1,5,f)!=5 || memcmp(m,"OBPE1",5)) { fclose(f); return 0; }
    if (fread(&vs,4,1,f)!=1 || fread(&nm,4,1,f)!=1) { fclose(f); return 0; }
    t->vs=vs; t->nm=nm; t->ma=malloc(nm*2); t->mb=malloc(nm*2);
    for (uint32_t k=0;k<nm;k++){ if(fread(&t->ma[k],2,1,f)!=1||fread(&t->mb[k],2,1,f)!=1){fclose(f);return 0;} }
    t->ex=malloc(vs*sizeof(uint8_t*)); t->el=malloc(vs*sizeof(uint16_t));
    for (uint32_t i=0;i<vs;i++){ uint16_t l; if(fread(&l,2,1,f)!=1){fclose(f);return 0;}
        t->el[i]=l; t->ex[i]=malloc(l?l:1); if(l&&fread(t->ex[i],1,l,f)!=l){fclose(f);return 0;} }
    fclose(f); return 1;
}
static size_t tok_encode(const Tok *t, const char *s, size_t sl, sym_t *o, size_t cap) {
    size_t n=0; for (size_t i=0;i<sl&&n<cap;i++) o[n++]=(unsigned char)s[i];
    for (int k=0;k<t->nm;k++){ uint16_t a=t->ma[k],b=t->mb[k],c=(uint16_t)(256+k); size_t j=0;
        for (size_t i=0;i<n;i++){ if(i+1<n&&o[i]==a&&o[i+1]==b){o[j++]=c;i++;} else o[j++]=o[i]; } n=j; }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s <ckpt> <vocab> <n> [temp] [seed]\n", argv[0]); return 1; }
    const char *ckpt=argv[1], *vpath=argv[2];
    int nout = atoi(argv[3]);
    float temp = (argc>4)?(float)atof(argv[4]):0.7f;
    const char *seed = (argc>5)?argv[5]:"static void ";

    Params *p = malloc(sizeof(Params));
    if (!load_ckpt(ckpt,p)) { fprintf(stderr,"cannot load %s\n",ckpt); return 1; }
    Tok tok; if (!tok_load(vpath,&tok)) { fprintf(stderr,"cannot load %s\n",vpath); return 1; }
    if (tok.vs != V) { fprintf(stderr,"vocab %d != build V=%d\n",tok.vs,V); return 1; }

    sym_t ctx[B], enc[8192];
    size_t m = tok_encode(&tok, seed, strlen(seed), enc, 8192);
    for (int t=0;t<B;t++){ long idx=(long)m-B+t; ctx[t]=(idx>=0)?enc[idx]:(sym_t)' '; }
    fputs(seed, stdout);

    rng ^= (uint64_t)time(NULL)*0x2545F4914F6CDD1DULL;
    float *probs = malloc(V*sizeof(float));
    for (int i=0;i<nout;i++){
        forward(p, ctx, probs);
        float mx=-1e30f;
        for (int k=0;k<V;k++){ float lp=k_logf(probs[k]+1e-12f)/temp; probs[k]=lp; if(lp>mx)mx=lp; }
        float s=0; for(int k=0;k<V;k++){ probs[k]=k_expf(probs[k]-mx); s+=probs[k]; }
        float r=frand()*s, c=0; int nx=V-1;
        for (int k=0;k<V;k++){ c+=probs[k]; if(r<=c){nx=k;break;} }
        fwrite(tok.ex[nx],1,tok.el[nx],stdout);
        memmove(ctx,ctx+1,(B-1)*sizeof(sym_t)); ctx[B-1]=(sym_t)nx;
    }
    fputc('\n', stdout);
    return 0;
}
