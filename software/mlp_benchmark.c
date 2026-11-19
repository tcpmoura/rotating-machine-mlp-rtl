/* Bit-exact C inference and Windows QPC benchmark. No external dependencies. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

typedef struct { int16_t h[8]; int64_t o[5]; } Trace;
typedef struct { int16_t x[4],h[8]; int64_t o[5]; unsigned cls; int score; } Vector;
static int16_t w[85];
static int input_frac,w1_frac,w2_frac,hidden_frac;
static volatile uint64_t sink;

static int64_t arithmetic_shift(int64_t x,int shift) {
    if(shift<0) return x*((int64_t)1<<(-shift));
    if(x>=0) return x>>shift;
    return -(((-x)+(((int64_t)1<<shift)-1))>>shift);
}
__attribute__((noinline,noipa))
static uint32_t infer_one(const int16_t x[4],Trace *trace) {
    int16_t h[8];int64_t out[5];
    int hs=input_frac+w1_frac-hidden_frac;
    for(int j=0;j<8;j++) {
        int64_t a=(int64_t)w[32+j]*((int64_t)1<<input_frac);
        for(int k=0;k<4;k++) a+=(int64_t)x[k]*w[4*j+k];
        a=a<0?0:(a+((int64_t)1<<(hs-1)))>>hs;
        h[j]=(int16_t)(a>32767?32767:a);
    }
    for(int j=0;j<5;j++) {
        int64_t a=(int64_t)w[80+j]*((int64_t)1<<hidden_frac);
        for(int k=0;k<8;k++) a+=(int64_t)h[k]*w[40+8*j+k];
        out[j]=a;
    }
    unsigned cls=0;
    for(unsigned j=1;j<5;j++)if(out[j]>out[cls])cls=j;
    int64_t score=arithmetic_shift(out[cls],hidden_frac+w2_frac-5);
    score=score>127?127:score< -128?-128:score;
    unsigned alarm=cls==0?8:cls<3?4:3;
    if(trace) {
        for(int j=0;j<8;j++)trace->h[j]=h[j];
        for(int j=0;j<5;j++)trace->o[j]=out[j];
    }
    return (uint32_t)cls|((uint32_t)(score&255)<<3)|(alarm<<11);
}
static int64_t signed_word(uint64_t v,int bits) {
    return (v&((uint64_t)1<<(bits-1)))?(int64_t)v-((int64_t)1<<bits):(int64_t)v;
}
static size_t load_vectors(const char *path,Vector **out) {
    FILE *f=fopen(path,"r");if(!f){perror(path);exit(2);}
    Vector *v=calloc(28080,sizeof *v);if(!v)exit(2);
    size_t n=0;uint64_t a[19];
    for(;;) {
        if(fscanf(f,"%" SCNx64,&a[0])!=1)break;
        if(n>=28080)exit(3);
        for(int k=1;k<19;k++)if(fscanf(f,"%" SCNx64,&a[k])!=1)exit(3);
        for(int k=0;k<4;k++)v[n].x[k]=(int16_t)signed_word(a[k],16);
        for(int k=0;k<8;k++)v[n].h[k]=(int16_t)signed_word(a[k+4],16);
        for(int k=0;k<5;k++)v[n].o[k]=signed_word(a[k+12],36);
        v[n].cls=(unsigned)a[17];v[n].score=(int)signed_word(a[18],8);n++;
    }
    fclose(f);*out=v;return n;
}
static void verify(const Vector *v,size_t n) {
    for(size_t i=0;i<n;i++) {
        Trace t;uint32_t p=infer_one(v[i].x,&t);
        uint32_t expected=v[i].cls|((unsigned)(v[i].score&255)<<3)|((v[i].cls==0?8:v[i].cls<3?4:3)<<11);
        if(p!=expected){fprintf(stderr,"class/score/alarms mismatch %zu\n",i);exit(4);}
        for(int j=0;j<8;j++)if(t.h[j]!=v[i].h[j]){fprintf(stderr,"hidden mismatch\n");exit(4);}
        for(int j=0;j<5;j++)if(t.o[j]!=v[i].o[j]){fprintf(stderr,"accumulator mismatch\n");exit(4);}
    }
}
static uint64_t ticks(void){LARGE_INTEGER t;QueryPerformanceCounter(&t);return (uint64_t)t.QuadPart;}
static uint64_t block(const int16_t (*x)[4],size_t n,size_t reps,int chain,uint64_t *sum) {
    uint64_t checksum=0;size_t index=0;uint64_t start=ticks();
    for(size_t r=0;r<reps;r++)for(size_t i=0;i<n;i++) {
        uint32_t p=infer_one(x[chain?index:i],NULL);checksum+=p;
        if(chain)index=(index+1+(p&255))%n;
    }
    uint64_t elapsed=ticks()-start;*sum=checksum;sink^=checksum;return elapsed;
}
int main(int argc,char **argv) {
    if(argc!=8){fprintf(stderr,"usage: exe rom vectors stress input_frac w1_frac w2_frac hidden_frac\n");return 2;}
    input_frac=atoi(argv[4]);w1_frac=atoi(argv[5]);w2_frac=atoi(argv[6]);hidden_frac=atoi(argv[7]);
    if(input_frac+w1_frac-hidden_frac<1)return 2;
    FILE *f=fopen(argv[1],"r");if(!f)return 2;
    for(int i=0;i<85;i++){unsigned v;if(fscanf(f,"%x",&v)!=1)return 2;w[i]=(int16_t)signed_word(v,16);}fclose(f);
    Vector *v,*stress;size_t n=load_vectors(argv[2],&v),ns=load_vectors(argv[3],&stress);
    verify(v,n);verify(stress,ns);free(stress);
    int16_t (*x)[4]=malloc(n*sizeof *x);if(!x)return 2;
    for(size_t i=0;i<n;i++)for(int k=0;k<4;k++)x[i][k]=v[i].x[k];
    free(v);
    DWORD_PTR affinity=SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1);
    LARGE_INTEGER freq;QueryPerformanceFrequency(&freq);
    uint64_t timer_min=UINT64_MAX;
    for(int i=0;i<1000;i++){uint64_t t=ticks();uint64_t dt=ticks()-t;if(dt<timer_min)timer_min=dt;}
    printf("META,verified,%zu,stress,%zu,qpc_hz,%lld,pointer_bits,%zu,affinity_applied,%d,timer_min_ticks,%" PRIu64 "\n",n,ns,(long long)freq.QuadPart,8*sizeof(void*),affinity!=0,timer_min);
    size_t reps[2];uint64_t checksum;
    for(int mode=0;mode<2;mode++) {
        block(x,n,4,mode,&checksum);
        uint64_t t=block(x,n,8,mode,&checksum);
        double estimate=(double)t/(double)freq.QuadPart;
        reps[mode]=(size_t)(.030*8/estimate)+1;if(reps[mode]<1)reps[mode]=1;
        if(reps[mode]>10000)reps[mode]=10000;
    }
    printf("trial,mode,inferences,ticks,ns_per_inference,checksum\n");
    for(int trial=0;trial<31;trial++)for(int order=0;order<2;order++) {
        int mode=(trial+order)%2;uint64_t sum;
        uint64_t dt=block(x,n,reps[mode],mode,&sum);
        printf("%d,%s,%zu,%" PRIu64 ",%.9f,%" PRIu64 "\n",trial,mode?"dependent_serial":"continuous",n*reps[mode],dt,1e9*(double)dt/(double)freq.QuadPart/(double)(n*reps[mode]),sum);
    }
    printf("SINK,%" PRIu64 "\n",sink);free(x);return 0;
}
