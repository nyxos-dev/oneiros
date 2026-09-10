/*
 * ngen_xln.c - Oneiros RMSNorm-transformer inference, self-contained for NyxOS.
 *
 * Runs the RMSNorm champion (ONEIROS4). Own expf/logf/tanhf/sqrtf, no libm; only
 * malloc/fopen/fread from NyxOS's libc. Verified byte-identical to the libm
 * `xformer_ln`. The in-OS demo path for the current best model.
 *
 * Build:  gcc -O2 -DV=2048 -o ngen_xln src/ngen_xln.c            (no -lm)
 * Run:    ./ngen_xln data/xformer_ln_2048.bin data/vocab2048.bin 60 0.7 "static void "
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifndef V
#define V 2048
#endif
#ifndef E
#define E 64
#endif
#define D E
#ifndef B
#define B 8
#endif
#ifndef NH
#define NH 4
#endif
#define DH (D/NH)
#ifndef FF
#define FF 256
#endif
#ifndef HID
#define HID 256
#endif
#define RMS_EPS 1e-5f

typedef uint16_t sym_t;

/* Params: byte-for-byte the same order/layout as xformer_ln.c */
typedef struct {
    float C[V*D], P[B*D];
    float ga[D], gf[D], gh[D];
    float Wq[D*D], Wk[D*D], Wv[D*D], Wo[D*D];
    float Wf1[FF*D], bf1[FF];
    float Wf2[D*FF], bf2[D];
    float Wh[HID*D], bh[HID];
    float Wout[V*HID], bout[V];
} Params;

/* own float math (no libm) */
static float k_expf(float x){
    if (x>88.0f) return 3.0e38f;
    if (x<-88.0f) return 0.0f;
    const float LOG2E=1.44269504f, LN2=0.6931471805f;
    int n=(int)(x*LOG2E+(x>=0?0.5f:-0.5f)); float r=x-(float)n*LN2;
    float p=1.0f+r*(1.0f+r*(0.5f+r*(0.16666667f+r*(0.041666668f+r*0.008333334f))));
    union{float f;int32_t i;}u; u.i=(int32_t)((n+127)<<23); return p*u.f;
}
static float k_logf(float x){
    if (x<=0.0f) return -88.0f;
    union{float f;int32_t i;}u; u.f=x; int e=((u.i>>23)&0xFF)-127;
    u.i=(u.i&0x807FFFFF)|0x3F800000; float m=u.f,t=(m-1.0f)/(m+1.0f),t2=t*t;
    float s=t*(2.0f+t2*(0.6666667f+t2*(0.4f+t2*0.2857143f))); return (float)e*0.6931471805f+s;
}
static float k_tanhf(float x){
    if (x>15.0f) return 1.0f;
    if (x<-15.0f) return -1.0f;
    float e=k_expf(2.0f*x); return (e-1.0f)/(e+1.0f);
}
static float k_sqrtf(float x){ if(x<=0)return 0; float g=x; for(int i=0;i<24;i++) g=0.5f*(g+x/g); return g; }

static uint64_t rng=0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float frand(void){ return (float)(rnd()/4294967296.0); }

/* RMSNorm: y = g * x / rms(x) */
static void rms(const float *x, const float *g, float *y){
    float ss=0; for(int i=0;i<D;i++) ss+=x[i]*x[i];
    float irms=1.0f/k_sqrtf(ss/(float)D+RMS_EPS);
    for(int i=0;i<D;i++) y[i]=g[i]*x[i]*irms;
}

static void forward(const Params *p, const sym_t *tok, float *probs){
    const int L=B-1; const float scale=1.0f/k_sqrtf((float)DH);
    float xt[B][D], xn[B][D], q[D], k[B][D], v[B][D], att[NH][B];
    float ctx[D], attn[D], r1[D], rn[D], g[FF], r2[D], rh[D], hid[HID];
    for (int t=0;t<B;t++) for (int d=0;d<D;d++) xt[t][d]=p->C[(int)tok[t]*D+d]+p->P[t*D+d];
    for (int t=0;t<B;t++) rms(xt[t], p->ga, xn[t]);
    for (int a=0;a<D;a++){ float s=0; for(int b=0;b<D;b++) s+=p->Wq[a*D+b]*xn[L][b]; q[a]=s; }
    for (int t=0;t<B;t++) for (int a=0;a<D;a++){ float ks=0,vs=0;
        for (int b=0;b<D;b++){ ks+=p->Wk[a*D+b]*xn[t][b]; vs+=p->Wv[a*D+b]*xn[t][b]; }
        k[t][a]=ks; v[t][a]=vs; }
    for (int h=0;h<NH;h++){ int o=h*DH; float sc[B],mx=-1e30f;
        for (int t=0;t<B;t++){ float s=0; for(int i=0;i<DH;i++) s+=q[o+i]*k[t][o+i]; s*=scale; sc[t]=s; if(s>mx)mx=s; }
        float sum=0; for(int t=0;t<B;t++){ float e=k_expf(sc[t]-mx); att[h][t]=e; sum+=e; }
        for (int t=0;t<B;t++) att[h][t]/=sum;
        for (int i=0;i<DH;i++){ float c=0; for(int t=0;t<B;t++) c+=att[h][t]*v[t][o+i]; ctx[o+i]=c; }
    }
    for (int e=0;e<D;e++){ float s=0; for(int d=0;d<D;d++) s+=p->Wo[e*D+d]*ctx[d]; attn[e]=s; }
    for (int d=0;d<D;d++) r1[d]=xt[L][d]+attn[d];
    rms(r1, p->gf, rn);
    for (int j=0;j<FF;j++){ float s=p->bf1[j]; for(int d=0;d<D;d++) s+=p->Wf1[j*D+d]*rn[d]; g[j]=s; }
    for (int d=0;d<D;d++){ float s=p->bf2[d]; for(int j=0;j<FF;j++){ float r=g[j]>0?g[j]:0; s+=p->Wf2[d*FF+j]*r; } r2[d]=r1[d]+s; }
    rms(r2, p->gh, rh);
    for (int m=0;m<HID;m++){ float z=p->bh[m]; for(int d=0;d<D;d++) z+=p->Wh[m*D+d]*rh[d]; hid[m]=k_tanhf(z); }
    float mx=-1e30f;
    for (int c=0;c<V;c++){ float z=p->bout[c]; for(int m=0;m<HID;m++) z+=p->Wout[c*HID+m]*hid[m]; probs[c]=z; if(z>mx)mx=z; }
    float s=0; for(int c=0;c<V;c++){ probs[c]=k_expf(probs[c]-mx); s+=probs[c]; }
    for (int c=0;c<V;c++) probs[c]/=s;
}

static int load_ckpt(const char *path, Params *p){
    FILE *f=fopen(path,"rb"); if(!f) return 0; char m[8]; int32_t dims[6];
    if (fread(m,1,8,f)!=8||memcmp(m,"ONEIROS4",8)){ fclose(f); return 0; }
    if (fread(dims,sizeof(dims),1,f)!=1){ fclose(f); return 0; }
    if (dims[0]!=V||dims[1]!=E||dims[2]!=B||dims[3]!=HID||dims[4]!=NH||dims[5]!=FF){
        fprintf(stderr,"ckpt dims mismatch\n"); fclose(f); return 0; }
    int ok=fread(p,sizeof(Params),1,f)==1; fclose(f); return ok;
}

typedef struct { int vs,nm; uint16_t *ma,*mb; uint8_t **ex; uint16_t *el; } Tok;
static int tok_load(const char *path, Tok *t){
    FILE *f=fopen(path,"rb"); if(!f) return 0; char m[5]; uint32_t vs,nm;
    if (fread(m,1,5,f)!=5||memcmp(m,"OBPE1",5)){ fclose(f); return 0; }
    if (fread(&vs,4,1,f)!=1||fread(&nm,4,1,f)!=1){ fclose(f); return 0; }
    t->vs=vs;t->nm=nm;t->ma=malloc(nm*2);t->mb=malloc(nm*2);
    for (uint32_t k=0;k<nm;k++){ if(fread(&t->ma[k],2,1,f)!=1||fread(&t->mb[k],2,1,f)!=1){fclose(f);return 0;} }
    t->ex=malloc(vs*sizeof(uint8_t*)); t->el=malloc(vs*sizeof(uint16_t));
    for (uint32_t i=0;i<vs;i++){ uint16_t l; if(fread(&l,2,1,f)!=1){fclose(f);return 0;}
        t->el[i]=l; t->ex[i]=malloc(l?l:1); if(l&&fread(t->ex[i],1,l,f)!=l){fclose(f);return 0;} }
    fclose(f); return 1;
}
static size_t tok_encode(const Tok *t, const char *s, size_t sl, sym_t *o, size_t cap){
    size_t n=0; for(size_t i=0;i<sl&&n<cap;i++) o[n++]=(unsigned char)s[i];
    for (int k=0;k<t->nm;k++){ uint16_t a=t->ma[k],b=t->mb[k],c=(uint16_t)(256+k); size_t j=0;
        for (size_t i=0;i<n;i++){ if(i+1<n&&o[i]==a&&o[i+1]==b){o[j++]=c;i++;} else o[j++]=o[i]; } n=j; }
    return n;
}

int main(int argc, char **argv){
    if (argc<4){ fprintf(stderr,"usage: %s <ckpt> <vocab> <n> [temp] [seed]\n",argv[0]); return 1; }
    int nout=atoi(argv[3]); float temp=(argc>4)?(float)atof(argv[4]):0.7f;
    const char *seed=(argc>5)?argv[5]:"static void ";
    Params *p=malloc(sizeof(Params)); if(!load_ckpt(argv[1],p)){ fprintf(stderr,"cannot load %s\n",argv[1]); return 1; }
    Tok tok; if(!tok_load(argv[2],&tok)){ fprintf(stderr,"cannot load %s\n",argv[2]); return 1; }
    if (tok.vs!=V){ fprintf(stderr,"vocab %d != V=%d\n",tok.vs,V); return 1; }
    float *probs=malloc(V*sizeof(float));
    sym_t ctx[B], enc[8192]; size_t m=tok_encode(&tok,seed,strlen(seed),enc,8192);
    for (int t=0;t<B;t++){ long idx=(long)m-B+t; ctx[t]=(idx>=0)?enc[idx]:(sym_t)' '; }
    fputs(seed,stdout);
    rng^=(uint64_t)time(NULL)*0x2545F4914F6CDD1DULL;
    for (int i=0;i<nout;i++){
        forward(p,ctx,probs);
        float mx=-1e30f; for(int c=0;c<V;c++){ float lp=k_logf(probs[c]+1e-12f)/temp; probs[c]=lp; if(lp>mx)mx=lp; }
        float s=0; for(int c=0;c<V;c++){ probs[c]=k_expf(probs[c]-mx); s+=probs[c]; }
        float r=frand()*s,cc=0; int nx=V-1; for(int c=0;c<V;c++){ cc+=probs[c]; if(r<=cc){nx=c;break;} }
        fwrite(tok.ex[nx],1,tok.el[nx],stdout);
        memmove(ctx,ctx+1,(B-1)*sizeof(sym_t)); ctx[B-1]=(sym_t)nx;
    }
    fputc('\n',stdout); return 0;
}
