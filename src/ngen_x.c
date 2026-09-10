/*
 * ngen_x.c - Oneiros TRANSFORMER inference, self-contained for inside NyxOS.
 *
 * Same idea as ngen.c but runs the transformer champion (ONEIROS3): multi-head
 * attention + FFN + residual + D->HID->V head. Uses only malloc/fopen/fread +
 * its OWN expf/logf/tanhf (no libm; NyxOS libm lacks exp/log/tanh, kernel is
 * -mno-sse). Links with no -lm; verified byte-identical to the libm `xformer`.
 *
 * Build:  gcc -O2 -DV=2048 -o ngen_x src/ngen_x.c            (no -lm)
 * Run:    ./ngen_x data/xformer_2048.bin data/vocab2048.bin 60 0.7 "static void "
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

typedef uint16_t sym_t;

typedef struct {
    float C[V*D], P[B*D];
    float Wq[D*D], Wk[D*D], Wv[D*D], Wo[D*D];
    float Wf1[FF*D], bf1[FF];
    float Wf2[D*FF], bf2[D];
    float Wh[HID*D], bh[HID];
    float Wout[V*HID], bout[V];
} Params;

typedef struct {
    float xt[B][D], q[D], k[B][D], v[B][D];
    float att[NH][B], ctx[D], attn[D], r1[D];
    float g[FF], r2[D], hid[HID];
    float *probs;
} Act;

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
static float k_sqrtf(float x){ if(x<=0)return 0; float g=x; for(int i=0;i<20;i++) g=0.5f*(g+x/g); return g; }

static uint64_t rng=0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float frand(void){ return (float)(rnd()/4294967296.0); }

static void forward(const Params *p, const sym_t *tok, Act *A){
    const int L=B-1; const float scale=1.0f/k_sqrtf((float)DH);
    for (int t=0;t<B;t++) for (int d=0;d<D;d++) A->xt[t][d]=p->C[(int)tok[t]*D+d]+p->P[t*D+d];
    for (int a=0;a<D;a++){ float s=0; for(int b=0;b<D;b++) s+=p->Wq[a*D+b]*A->xt[L][b]; A->q[a]=s; }
    for (int t=0;t<B;t++) for (int a=0;a<D;a++){ float ks=0,vs=0;
        for (int b=0;b<D;b++){ ks+=p->Wk[a*D+b]*A->xt[t][b]; vs+=p->Wv[a*D+b]*A->xt[t][b]; }
        A->k[t][a]=ks; A->v[t][a]=vs; }
    for (int h=0;h<NH;h++){ int o=h*DH; float sc[B],mx=-1e30f;
        for (int t=0;t<B;t++){ float s=0; for(int i=0;i<DH;i++) s+=A->q[o+i]*A->k[t][o+i]; s*=scale; sc[t]=s; if(s>mx)mx=s; }
        float sum=0; for(int t=0;t<B;t++){ float e=k_expf(sc[t]-mx); A->att[h][t]=e; sum+=e; }
        for (int t=0;t<B;t++) A->att[h][t]/=sum;
        for (int i=0;i<DH;i++){ float c=0; for(int t=0;t<B;t++) c+=A->att[h][t]*A->v[t][o+i]; A->ctx[o+i]=c; }
    }
    for (int e=0;e<D;e++){ float s=0; for(int d=0;d<D;d++) s+=p->Wo[e*D+d]*A->ctx[d]; A->attn[e]=s; }
    for (int d=0;d<D;d++) A->r1[d]=A->xt[L][d]+A->attn[d];
    for (int j=0;j<FF;j++){ float s=p->bf1[j]; for(int d=0;d<D;d++) s+=p->Wf1[j*D+d]*A->r1[d]; A->g[j]=s; }
    for (int d=0;d<D;d++){ float s=p->bf2[d]; for(int j=0;j<FF;j++){ float r=A->g[j]>0?A->g[j]:0; s+=p->Wf2[d*FF+j]*r; } A->r2[d]=A->r1[d]+s; }
    for (int m=0;m<HID;m++){ float z=p->bh[m]; for(int d=0;d<D;d++) z+=p->Wh[m*D+d]*A->r2[d]; A->hid[m]=k_tanhf(z); }
    float mx=-1e30f;
    for (int k=0;k<V;k++){ float z=p->bout[k]; for(int m=0;m<HID;m++) z+=p->Wout[k*HID+m]*A->hid[m]; A->probs[k]=z; if(z>mx)mx=z; }
    float s=0; for(int k=0;k<V;k++){ A->probs[k]=k_expf(A->probs[k]-mx); s+=A->probs[k]; }
    for (int k=0;k<V;k++) A->probs[k]/=s;
}

static int load_ckpt(const char *path, Params *p){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    char m[8]; int32_t dims[6];
    if (fread(m,1,8,f)!=8||memcmp(m,"ONEIROS3",8)){ fclose(f); return 0; }
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
    Act A; A.probs=malloc(V*sizeof(float));
    sym_t ctx[B], enc[8192]; size_t m=tok_encode(&tok,seed,strlen(seed),enc,8192);
    for (int t=0;t<B;t++){ long idx=(long)m-B+t; ctx[t]=(idx>=0)?enc[idx]:(sym_t)' '; }
    fputs(seed,stdout);
    rng^=(uint64_t)time(NULL)*0x2545F4914F6CDD1DULL;
    for (int i=0;i<nout;i++){
        forward(p,ctx,&A);
        float mx=-1e30f; for(int k=0;k<V;k++){ float lp=k_logf(A.probs[k]+1e-12f)/temp; A.probs[k]=lp; if(lp>mx)mx=lp; }
        float s=0; for(int k=0;k<V;k++){ A.probs[k]=k_expf(A.probs[k]-mx); s+=A.probs[k]; }
        float r=frand()*s,c=0; int nx=V-1; for(int k=0;k<V;k++){ c+=A.probs[k]; if(r<=c){nx=k;break;} }
        fwrite(tok.ex[nx],1,tok.el[nx],stdout);
        memmove(ctx,ctx+1,(B-1)*sizeof(sym_t)); ctx[B-1]=(sym_t)nx;
    }
    fputc('\n',stdout); return 0;
}
