/*
 * xformer.c - Oneiros, a real (minimal) transformer block, from scratch.
 *
 * The honest "does attention actually pay off?" test. One transformer block:
 *   x_t = emb(tok_t) + pos_t                                 (t=0..B-1, dim D)
 *   multi-head self-attention (NH heads), query at the last position
 *     -> concat heads -> Wo -> attn ; r1 = x_{B-1} + attn    (residual)
 *   FFN: f = Wf2 relu(Wf1 r1 + bf1) + bf2 ; r2 = r1 + f       (residual)
 *   head: h = tanh(Wh r2 + bh) ; logits = Wout h + bout       (D->HID->V, matches
 *                                                              the MLP output head)
 * Hand-written forward + backprop + Adagrad; verified by `gradcheck`.
 * Trains on bytes (V=256) or BPE tokens (-DV=2048, .tok). No ML frameworks.
 *
 * Build:  gcc -O2 -march=native -DV=2048 -o xformer src/xformer.c -lm
 * Verbs:  train <corpus.tok> <steps> <out.bin> [resume] | sample ... | gradcheck
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifndef V
#define V   256
#endif
#ifndef E
#define E   64            /* model dim D                            */
#endif
#define D   E
#ifndef B
#define B   8             /* context window                        */
#endif
#ifndef NH
#define NH  4             /* attention heads                       */
#endif
#define DH  (D / NH)      /* head dim                              */
#ifndef FF
#define FF  256           /* FFN inner dim                         */
#endif
#ifndef HID
#define HID 256           /* output-head hidden (matches the MLP)  */
#endif
#define BATCH 32
#define VAL_FRAC 0.10f

typedef uint16_t sym_t;

typedef struct {
    float C[V*D];
    float P[B*D];
    float Wq[D*D], Wk[D*D], Wv[D*D], Wo[D*D];
    float Wf1[FF*D], bf1[FF];
    float Wf2[D*FF], bf2[D];
    float Wh[HID*D], bh[HID];
    float Wout[V*HID], bout[V];
} Params;

typedef struct { Params g2; } Adagrad;

typedef struct {
    float xt[B][D];
    float q[D], k[B][D], v[B][D];
    float att[NH][B];
    float ctx[D], attn[D], r1[D];
    float g[FF], r2[D], hid[HID];
    float probs[V];
} Act;

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float frand(void){ return (float)(rnd()/4294967296.0); }
static float gauss(float sd){ float u1=frand()+1e-9f,u2=frand(); return sd*sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }

static void init_params(Params *p){
    for (int i=0;i<V*D;i++) p->C[i]=gauss(0.02f);
    for (int i=0;i<B*D;i++) p->P[i]=gauss(0.02f);
    float sd=1.0f/sqrtf((float)D), sf=1.0f/sqrtf((float)FF), sh=1.0f/sqrtf((float)HID);
    for (int i=0;i<D*D;i++){ p->Wq[i]=gauss(sd); p->Wk[i]=gauss(sd); p->Wv[i]=gauss(sd); p->Wo[i]=gauss(sd); }
    for (int i=0;i<FF*D;i++) p->Wf1[i]=gauss(sd);
    for (int i=0;i<FF;i++)   p->bf1[i]=0.0f;
    for (int i=0;i<D*FF;i++) p->Wf2[i]=gauss(sf);
    for (int i=0;i<D;i++)    p->bf2[i]=0.0f;
    for (int i=0;i<HID*D;i++)p->Wh[i]=gauss(sd);
    for (int i=0;i<HID;i++)  p->bh[i]=0.0f;
    for (int i=0;i<V*HID;i++)p->Wout[i]=gauss(sh);
    for (int i=0;i<V;i++)    p->bout[i]=0.0f;
}

static float forward(const Params *p, const sym_t *tok, int target, Act *A){
    const int L=B-1; const float scale=1.0f/sqrtf((float)DH);
    for (int t=0;t<B;t++) for (int d=0;d<D;d++) A->xt[t][d]=p->C[(int)tok[t]*D+d]+p->P[t*D+d];
    for (int a=0;a<D;a++){ float s=0; for (int b=0;b<D;b++) s+=p->Wq[a*D+b]*A->xt[L][b]; A->q[a]=s; }
    for (int t=0;t<B;t++) for (int a=0;a<D;a++){ float ks=0,vs=0;
        for (int b=0;b<D;b++){ ks+=p->Wk[a*D+b]*A->xt[t][b]; vs+=p->Wv[a*D+b]*A->xt[t][b]; }
        A->k[t][a]=ks; A->v[t][a]=vs; }
    for (int h=0;h<NH;h++){ int o=h*DH; float sc[B],mx=-1e30f;
        for (int t=0;t<B;t++){ float s=0; for (int i=0;i<DH;i++) s+=A->q[o+i]*A->k[t][o+i]; s*=scale; sc[t]=s; if(s>mx)mx=s; }
        float sum=0; for (int t=0;t<B;t++){ float e=expf(sc[t]-mx); A->att[h][t]=e; sum+=e; }
        for (int t=0;t<B;t++) A->att[h][t]/=sum;
        for (int i=0;i<DH;i++){ float c=0; for (int t=0;t<B;t++) c+=A->att[h][t]*A->v[t][o+i]; A->ctx[o+i]=c; }
    }
    for (int e=0;e<D;e++){ float s=0; for (int d=0;d<D;d++) s+=p->Wo[e*D+d]*A->ctx[d]; A->attn[e]=s; }
    for (int d=0;d<D;d++) A->r1[d]=A->xt[L][d]+A->attn[d];
    for (int j=0;j<FF;j++){ float s=p->bf1[j]; for (int d=0;d<D;d++) s+=p->Wf1[j*D+d]*A->r1[d]; A->g[j]=s; }
    for (int d=0;d<D;d++){ float s=p->bf2[d]; for (int j=0;j<FF;j++){ float r=A->g[j]>0?A->g[j]:0; s+=p->Wf2[d*FF+j]*r; } A->r2[d]=A->r1[d]+s; }
    for (int m=0;m<HID;m++){ float z=p->bh[m]; for (int d=0;d<D;d++) z+=p->Wh[m*D+d]*A->r2[d]; A->hid[m]=tanhf(z); }
    float mx=-1e30f;
    for (int k=0;k<V;k++){ float z=p->bout[k]; for (int m=0;m<HID;m++) z+=p->Wout[k*HID+m]*A->hid[m]; A->probs[k]=z; if(z>mx)mx=z; }
    float s=0; for (int k=0;k<V;k++){ A->probs[k]=expf(A->probs[k]-mx); s+=A->probs[k]; }
    for (int k=0;k<V;k++) A->probs[k]/=s;
    return -logf(A->probs[target]+1e-12f);
}

static void backward(const Params *p, const sym_t *tok, int target, const Act *A, Params *g){
    const int L=B-1; const float scale=1.0f/sqrtf((float)DH);
    float dz[V], dhid[HID];
    for (int k=0;k<V;k++) dz[k]=A->probs[k];
    dz[target]-=1.0f;
    for (int m=0;m<HID;m++) dhid[m]=0;
    for (int k=0;k<V;k++){ float d=dz[k]; if(d==0)continue; float *gw=g->Wout+k*HID; const float *w=p->Wout+k*HID;
        for (int m=0;m<HID;m++){ gw[m]+=d*A->hid[m]; dhid[m]+=d*w[m]; } g->bout[k]+=d; }
    float dr2[D]; for (int d=0;d<D;d++) dr2[d]=0;
    for (int m=0;m<HID;m++){ float draw=dhid[m]*(1.0f-A->hid[m]*A->hid[m]); float *gw=g->Wh+m*D; const float *w=p->Wh+m*D;
        for (int d=0;d<D;d++){ gw[d]+=draw*A->r2[d]; dr2[d]+=draw*w[d]; } g->bh[m]+=draw; }
    /* r2 = r1 + f */
    float dr1[D]; for (int d=0;d<D;d++) dr1[d]=dr2[d];
    float dg[FF]; for (int j=0;j<FF;j++) dg[j]=0;
    for (int d=0;d<D;d++){ float df=dr2[d]; g->bf2[d]+=df; float *gw=g->Wf2+d*FF; const float *w=p->Wf2+d*FF;
        for (int j=0;j<FF;j++){ float r=A->g[j]>0?A->g[j]:0; gw[j]+=df*r; if(A->g[j]>0) dg[j]+=df*w[j]; } }
    for (int j=0;j<FF;j++){ g->bf1[j]+=dg[j]; float *gw=g->Wf1+j*D; const float *w=p->Wf1+j*D;
        for (int d=0;d<D;d++){ gw[d]+=dg[j]*A->r1[d]; dr1[d]+=dg[j]*w[d]; } }
    /* r1 = x_L + attn */
    float dattn[D]; for (int d=0;d<D;d++) dattn[d]=dr1[d];
    float dx[B][D]; for (int t=0;t<B;t++) for (int d=0;d<D;d++) dx[t][d]=0;
    for (int d=0;d<D;d++) dx[L][d]+=dr1[d];               /* residual to x_L */
    /* attn = Wo ctx */
    float dctx[D]; for (int d=0;d<D;d++) dctx[d]=0;
    for (int e=0;e<D;e++){ float de=dattn[e]; float *gw=g->Wo+e*D; const float *w=p->Wo+e*D;
        for (int d=0;d<D;d++){ gw[d]+=de*A->ctx[d]; dctx[d]+=de*w[d]; } }
    /* per-head attention */
    float dq[D]; for (int a=0;a<D;a++) dq[a]=0;
    float dk[B][D], dv[B][D];
    for (int t=0;t<B;t++) for (int a=0;a<D;a++){ dk[t][a]=0; dv[t][a]=0; }
    for (int h=0;h<NH;h++){ int o=h*DH; float da[B];
        for (int t=0;t<B;t++) da[t]=0;
        for (int i=0;i<DH;i++) for (int t=0;t<B;t++){ da[t]+=dctx[o+i]*A->v[t][o+i]; dv[t][o+i]=A->att[h][t]*dctx[o+i]; }
        float sdot=0; for (int t=0;t<B;t++) sdot+=A->att[h][t]*da[t];
        float ds[B]; for (int t=0;t<B;t++) ds[t]=A->att[h][t]*(da[t]-sdot);
        for (int t=0;t<B;t++) for (int i=0;i<DH;i++){ dq[o+i]+=ds[t]*A->k[t][o+i]*scale; dk[t][o+i]=ds[t]*A->q[o+i]*scale; }
    }
    /* q = Wq x_L */
    for (int a=0;a<D;a++){ float dqa=dq[a]; float *gw=g->Wq+a*D; const float *w=p->Wq+a*D;
        for (int b=0;b<D;b++){ gw[b]+=dqa*A->xt[L][b]; dx[L][b]+=dqa*w[b]; } }
    /* k_t = Wk x_t ; v_t = Wv x_t */
    for (int t=0;t<B;t++) for (int a=0;a<D;a++){ float dkta=dk[t][a], dvta=dv[t][a];
        float *gwk=g->Wk+a*D; const float *wk=p->Wk+a*D; float *gwv=g->Wv+a*D; const float *wv=p->Wv+a*D;
        for (int b=0;b<D;b++){ gwk[b]+=dkta*A->xt[t][b]; dx[t][b]+=dkta*wk[b];
                               gwv[b]+=dvta*A->xt[t][b]; dx[t][b]+=dvta*wv[b]; } }
    /* x_t = C[tok]+P[t] */
    for (int t=0;t<B;t++) for (int d=0;d<D;d++){ g->C[(int)tok[t]*D+d]+=dx[t][d]; g->P[t*D+d]+=dx[t][d]; }
}

static void adagrad_step(Params *p, const Params *grad, Adagrad *ad, float lr){
    float *pf=(float*)p; const float *gf=(const float*)grad; float *af=(float*)&ad->g2;
    size_t n=sizeof(Params)/sizeof(float);
    for (size_t i=0;i<n;i++){ float gg=gf[i]; af[i]+=gg*gg; pf[i]-=lr*gg/(sqrtf(af[i])+1e-8f); }
}

static sym_t *load_syms(const char *path, size_t *out_n){
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    char magic[5]={0}; size_t got=fread(magic,1,5,f);
    if (got==5 && !memcmp(magic,"OTOK1",5)){
        uint32_t vs; uint64_t nt;
        if (fread(&vs,4,1,f)!=1||fread(&nt,8,1,f)!=1){fprintf(stderr,"bad tok\n");exit(1);}
        if ((int)vs!=V){fprintf(stderr,"tok vocab %u != V=%d\n",vs,V);exit(1);}
        sym_t *buf=malloc((size_t)nt*sizeof(sym_t)); size_t n=fread(buf,sizeof(sym_t),(size_t)nt,f);
        fclose(f); *out_n=n; fprintf(stderr,"loaded %zu tokens\n",n); return buf;
    }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *raw=malloc((size_t)sz); size_t n=fread(raw,1,(size_t)sz,f); fclose(f);
    sym_t *buf=malloc(n*sizeof(sym_t)); for(size_t i=0;i<n;i++) buf[i]=raw[i]; free(raw);
    *out_n=n; fprintf(stderr,"loaded %zu bytes\n",n); return buf;
}

static void save(const char *path, const Params *p){
    FILE *f=fopen(path,"wb"); if(!f){perror(path);exit(1);}
    int32_t dims[6]={V,E,B,HID,NH,FF};
    fwrite("ONEIROS3",1,8,f); fwrite(dims,sizeof(dims),1,f); fwrite(p,sizeof(Params),1,f); fclose(f);
}
static int load(const char *path, Params *p){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    char m[8]; int32_t dims[6];
    if (fread(m,1,8,f)!=8||memcmp(m,"ONEIROS3",8)){fclose(f);return 0;}
    if (fread(dims,sizeof(dims),1,f)!=1){fclose(f);return 0;}
    if (dims[0]!=V||dims[1]!=E||dims[2]!=B||dims[3]!=HID||dims[4]!=NH||dims[5]!=FF){
        fprintf(stderr,"ckpt dims mismatch\n"); fclose(f); return 0; }
    int ok=fread(p,sizeof(Params),1,f)==1; fclose(f); return ok;
}

static double eval_loss(const Params *p, const sym_t *text, size_t lo, size_t hi, int count){
    if (hi<lo+B+1) return 0;
    size_t span=hi-lo-B-1; if(!span)span=1;
    uint64_t saved=rng; double sum=0; Act A;
    for (int i=0;i<count;i++){ size_t pos=lo+(size_t)(((uint64_t)rnd()<<20^rnd())%span);
        sum+=forward(p,text+pos,(int)text[pos+B],&A); }
    rng=saved; return sum/count;
}

static void cmd_train(int argc, char **argv){
    const char *corpus=argv[2]; long steps=atol(argv[3]); const char *out=argv[4];
    const char *in=(argc>5)?argv[5]:NULL;
    size_t n; sym_t *text=load_syms(corpus,&n);
    if (n<(size_t)(B+2)){fprintf(stderr,"corpus too small\n");exit(1);}
    Params *p=malloc(sizeof(Params)),*grad=malloc(sizeof(Params)); Adagrad *ad=calloc(1,sizeof(Adagrad));
    if (in && load(in,p)) fprintf(stderr,"resumed\n"); else { init_params(p); fprintf(stderr,"fresh init\n"); }
    size_t train_end=(size_t)(n*(1.0-(double)VAL_FRAC)); if(train_end<(size_t)(B+2))train_end=n;
    size_t span=train_end-B-1;
    fprintf(stderr,"corpus %zu: train [0,%zu) val [%zu,%zu)  params %zu (D=%d NH=%d FF=%d HID=%d)\n",
            n,train_end,train_end,n,sizeof(Params)/sizeof(float),D,NH,FF,HID);
    double run=0; long runc=0; clock_t t0=clock();
    float base_lr=0.02f; { const char *e=getenv("ONEIROS_LR"); if(e) base_lr=(float)atof(e); }
    long warmup=2000;   /* batches of linear LR warmup (Adagrad cold-start would blow up the residual stream) */
    long batches=steps/BATCH; const float invb=1.0f/BATCH; size_t np=sizeof(Params)/sizeof(float); Act A;
    for (long bi=0;bi<batches;bi++){
        memset(grad,0,sizeof(Params)); double bl=0;
        for (int j=0;j<BATCH;j++){ size_t pos=(size_t)(((uint64_t)rnd()<<20^rnd())%span);
            bl+=forward(p,text+pos,text[pos+B],&A); backward(p,text+pos,text[pos+B],&A,grad); }
        float *gf=(float*)grad; for (size_t i=0;i<np;i++) gf[i]*=invb;
        float lr=base_lr*((bi<warmup)?((float)(bi+1)/(float)warmup):1.0f);
        adagrad_step(p,grad,ad,lr);
        run+=bl*invb; runc++; long done=(bi+1)*(long)BATCH;
        if (done%20000<BATCH){ double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
            double vl=eval_loss(p,text,train_end,n,2000);
            fprintf(stderr,"step %8ld  train %.4f  val %.4f  (%.0f samp/s)\n",done,run/runc,vl,done/(sec+1e-9));
            run=0; runc=0; }
    }
    save(out,p); double vl=eval_loss(p,text,train_end,n,20000);
    fprintf(stderr,"saved %s   final val loss (20k windows): %.4f\n",out,vl);
    free(text);free(p);free(grad);free(ad);
}

typedef struct { int vs,nm; uint16_t *ma,*mb; uint8_t **ex; uint16_t *el; } Tok;
static int tok_load(const char *path, Tok *t){
    FILE *f=fopen(path,"rb"); if(!f)return 0; char m[5]; uint32_t vs,nm;
    if (fread(m,1,5,f)!=5||memcmp(m,"OBPE1",5)){fclose(f);return 0;}
    if (fread(&vs,4,1,f)!=1||fread(&nm,4,1,f)!=1){fclose(f);return 0;}
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
static void cmd_sample(int argc, char **argv){
    const char *ckpt=argv[2]; int nn=atoi(argv[3]); float temp=(argc>4)?(float)atof(argv[4]):0.7f;
    const char *seed=(argc>5)?argv[5]:"static void "; const char *vp=(argc>6)?argv[6]:NULL;
    Params *p=malloc(sizeof(Params)); if(!load(ckpt,p)){fprintf(stderr,"cannot load %s\n",ckpt);exit(1);}
    Tok tok; int tm=0; if(vp){ if(!tok_load(vp,&tok)){fprintf(stderr,"cannot load vocab\n");exit(1);} tm=1; }
    sym_t ctx[B];
    if (tm){ sym_t enc[8192]; size_t m=tok_encode(&tok,seed,strlen(seed),enc,8192);
        for(int t=0;t<B;t++){ long idx=(long)m-B+t; ctx[t]=(idx>=0)?enc[idx]:(sym_t)' '; } }
    else { int sl=(int)strlen(seed); for(int t=0;t<B;t++){ int idx=sl-B+t; ctx[t]=(idx>=0)?(sym_t)(unsigned char)seed[idx]:(sym_t)' '; } }
    fputs(seed,stdout);
    Act A; rng^=(uint64_t)time(NULL)*0x2545F4914F6CDD1DULL;
    for (int i=0;i<nn;i++){ forward(p,ctx,0,&A);
        float mx=-1e30f; for(int k=0;k<V;k++){ float lp=logf(A.probs[k]+1e-12f)/temp; A.probs[k]=lp; if(lp>mx)mx=lp; }
        float s=0; for(int k=0;k<V;k++){ A.probs[k]=expf(A.probs[k]-mx); s+=A.probs[k]; }
        float r=frand()*s,c=0; int nx=V-1; for(int k=0;k<V;k++){ c+=A.probs[k]; if(r<=c){nx=k;break;} }
        if (tm) fwrite(tok.ex[nx],1,tok.el[nx],stdout); else putchar(nx);
        memmove(ctx,ctx+1,(B-1)*sizeof(sym_t)); ctx[B-1]=(sym_t)nx; }
    putchar('\n'); free(p);
}

static void cmd_gradcheck(void){
    Params *p=malloc(sizeof(Params)),*g=malloc(sizeof(Params)); Adagrad *ad=calloc(1,sizeof(Adagrad));
    init_params(p); Act A,tmp; sym_t ctx[B];
    for (int s=0;s<500;s++){ for(int t=0;t<B;t++) ctx[t]=(sym_t)(rnd()%V); int tg=(int)(rnd()%V);
        memset(g,0,sizeof(Params)); forward(p,ctx,tg,&A); backward(p,ctx,tg,&A,g); adagrad_step(p,g,ad,0.05f); }
    for (int t=0;t<B;t++) ctx[t]=(sym_t)(rnd()%V);
    int target=(int)(rnd()%V);
    memset(g,0,sizeof(Params)); forward(p,ctx,target,&A); backward(p,ctx,target,&A,g);
    float *pf=(float*)p; const float *gf=(const float*)g; size_t np=sizeof(Params)/sizeof(float);
    const float eps=2e-3f; int inf=0,good=0; double mr=0,mx=0;
    for (int n=0;n<300000 && inf<4000;n++){ size_t i=(size_t)(((uint64_t)rnd()<<20^rnd())%np);
        double ana=gf[i]; if (fabs(ana)<1e-3) continue;
        float sv=pf[i]; pf[i]=sv+eps; float lp=forward(p,ctx,target,&tmp);
        pf[i]=sv-eps; float lm=forward(p,ctx,target,&tmp); pf[i]=sv;
        double num=(lp-lm)/(2.0*eps); double rel=fabs(num-ana)/(fabs(num)+fabs(ana)+1e-9);
        mr+=rel; if(rel>mx)mx=rel; if(rel<0.05)good++; inf++; }
    printf("gradcheck: %d informative  mean rel %.5f  max %.5f  within5%%: %.1f%%\n",inf,mr/inf,mx,100.0*good/inf);
    printf("%s\n",(mr/inf<0.02)?"BACKPROP OK":"BACKPROP SUSPECT");
    free(p);free(g);free(ad);
}

int main(int argc, char **argv){
    if (argc>=5 && !strcmp(argv[1],"train")){ cmd_train(argc,argv); return 0; }
    if (argc>=4 && !strcmp(argv[1],"sample")){ cmd_sample(argc,argv); return 0; }
    if (argc>=2 && !strcmp(argv[1],"gradcheck")){ cmd_gradcheck(); return 0; }
    fprintf(stderr,"usage: %s train|sample|gradcheck ...\n",argv[0]); return 1;
}
