"""Width-parametric numerical calibration and integer arithmetic checks."""
from pathlib import Path
import csv, hashlib, itertools, json, pickle, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from threadpoolctl import threadpool_limits
from fixed_point import read_rom, infer
from calibrate import q as original_q

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/numeric_formats'
KEYS=('input_frac','w1_frac','w2_frac','hidden_frac')
FORMATS={'int8':(8,8,8,8),'int8_calibrated':(8,8,8,8),
         'int12':(12,12,12,12),'int16':(16,16,16,16),'mixed8_16':(8,8,16,8)}
threadpool_limits(1)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bits_for_bound(bound): return max(2,int(bound).bit_length()+1)

def quant(a,frac,width):
    return np.clip(np.rint(a*2.**frac),-2**(width-1),2**(width-1)-1).astype(np.int64)

def calibrate(arrays,width):
    bound=max(float(np.max(np.abs(a))) for a in arrays)
    return min(width-3,int(np.floor(np.log2((2**(width-1)-1)/bound)))) if bound else width-3

def parameters(model,sc,widths):
    _,w1,w2,_=widths
    return (quant(model.coefs_[0],sc['w1_frac'],w1),quant(model.intercepts_[0],sc['w1_frac'],w1),
            quant(model.coefs_[1],sc['w2_frac'],w2),quant(model.intercepts_[1],sc['w2_frac'],w2))

def evaluate(z,params,sc,widths):
    bx,bw1,bw2,bh=widths
    sx=2**sc['input_frac']; sh=2**sc['hidden_frac']
    shift=sc['input_frac']+sc['w1_frac']-sc['hidden_frac']
    assert shift>=1 and min(sc.values())>=0
    xraw=np.rint(z*sx); x=quant(z,sc['input_frac'],bx)
    w1,b1,w2,b2=params
    a=x@w1+b1*sx
    hr=(np.maximum(a,0)+(1<<(shift-1)))>>shift
    h=np.minimum(hr,2**(bh-1)-1)
    o=h@w2+b2*sh
    bound1=np.abs(x)@np.abs(w1)+np.abs(b1)*sx+(1<<(shift-1))
    bound2=np.abs(h)@np.abs(w2)+np.abs(b2)*sh
    bound=max(int(bound1.max()),int(bound2.max()))
    # Covers any representable inputs/parameters, including intermediate MAC sums.
    worst1=4*2**(bx-1)*2**(bw1-1)+2**(bw1-1)*sx+(1<<(shift-1))
    worst2=8*(2**(bh-1)-1)*2**(bw2-1)+2**(bw2-1)*sh
    worst=max(worst1,worst2)
    assert worst<2**62 and bound<=worst
    diag=dict(input_clipped_windows=int(np.any((xraw<-2**(bx-1))|(xraw>2**(bx-1)-1),axis=1).sum()),
              hidden_clipped_windows=int(np.any(hr>2**(bh-1)-1,axis=1).sum()),
              observed_accumulator_bound=bound,observed_accumulator_bits=bits_for_bound(bound),
              conservative_accumulator_bound=worst,conservative_accumulator_bits=bits_for_bound(worst),
              argmax_ties=int(((o==o.max(axis=1,keepdims=True)).sum(axis=1)>1).sum()))
    return o.argmax(1),o,h,diag

def metrics(y,p,pf):
    return dict(n=len(y),accuracy=float(accuracy_score(y,p)),
                macro_f1=float(f1_score(y,p,labels=range(5),average='macro',zero_division=0)),
                recall=recall_score(y,p,labels=range(5),average=None,zero_division=0).tolist(),
                agreement=float(np.mean(p==pf)),changed=int(np.sum(p!=pf)),
                harmed=int(np.sum((pf==y)&(p!=y))),corrected=int(np.sum((pf!=y)&(p==y))),
                both_wrong=int(np.sum((pf!=y)&(p!=y))))

def float_logits(m,z): return np.maximum(z@m.coefs_[0]+m.intercepts_[0],0)@m.coefs_[1]+m.intercepts_[1]

def adjust_int8(m,z,original):
    ref=float_logits(m,z); pf=ref.argmax(1); ref=ref-ref.mean(1,keepdims=True)
    trials=[]
    for delta in itertools.product([-1,0,1],repeat=4):
        sc={k:original[k]+d for k,d in zip(KEYS,delta)}
        if min(sc.values())<0 or sc['input_frac']+sc['w1_frac']-sc['hidden_frac']<1:continue
        p,o,_,_=evaluate(z,parameters(m,sc,FORMATS['int8']),sc,FORMATS['int8'])
        logits=o/2.**(sc['hidden_frac']+sc['w2_frac']);logits-=logits.mean(1,keepdims=True)
        trials.append(dict(scales=sc,disagreements=int(np.sum(p!=pf)),
                           centered_logit_mse=float(np.mean((logits-ref)**2)),distance=sum(abs(d) for d in delta)))
    best=min(trials,key=lambda t:(t['disagreements'],t['centered_logit_mse'],t['distance'],tuple(t['scales'][k] for k in KEYS)))
    return best['scales'],dict(selected=best,trials=trials,calibration_windows=len(z),uses_test=False)

def validate_arithmetic():
    # Independent Python-integer scalar oracle, including extreme signed operands.
    rng=np.random.default_rng(912)
    for widths in FORMATS.values():
        bx,bw1,bw2,bh=widths; sc=dict(input_frac=bx-3,w1_frac=bw1-3,w2_frac=bw2-3,hidden_frac=bh-3)
        x=rng.integers(-2**(bx-1),2**(bx-1),size=(12,4)); x[0]=-2**(bx-1);x[1]=2**(bx-1)-1
        pars=tuple(rng.integers(-2**(w-1),2**(w-1),size=shape) for w,shape in [(bw1,(4,8)),(bw1,(8,)),(bw2,(8,5)),(bw2,(5,))])
        pred,out,hidden,_=evaluate(x/2.**sc['input_frac'],pars,sc,widths)
        w1,b1,w2,b2=pars;shift=sc['input_frac']+sc['w1_frac']-sc['hidden_frac']
        for i,row in enumerate(x):
            h=[]
            for j in range(8):
                a=sum(int(row[k])*int(w1[k,j]) for k in range(4))+int(b1[j])*2**sc['input_frac']
                h.append(min((max(a,0)+2**(shift-1))//2**shift,2**(bh-1)-1))
            o=[sum(h[k]*int(w2[k,j]) for k in range(8))+int(b2[j])*2**sc['hidden_frac'] for j in range(5)]
            assert h==hidden[i].tolist() and o==out[i].tolist() and pred[i]==max(range(5),key=lambda j:o[j])
