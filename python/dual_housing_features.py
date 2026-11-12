"""Predeclared four-feature sets using both synchronous vibration housings."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/dual_housing';OUT.mkdir(parents=True,exist_ok=True)
SETS={name:list(range(i*4,i*4+4)) for i,name in enumerate([
    'control_absolute_a','harmonic_a','harmonic_ab','harmonic_bearing_ab','housing_ratios'])}

def power(w):
    n=w.shape[1];h=np.hanning(n);c=w-w.mean(axis=1,keepdims=True)
    p=np.abs(np.fft.rfft(c*h[None,:,None],axis=1))**2/(n*np.sum(h*h));p[:,1:-1]*=2
    return np.stack([p[:,:,:2].sum(axis=2),p[:,:,2:].sum(axis=2)],axis=2)

def form_features(p,f):
    e=np.stack([p[:,np.abs(f-v)<=15].sum(axis=1) for v in [50.17,100.34,150.51,179.43,272.07]],axis=1)
    low=p[:,(f>=35)&(f<350)].sum(axis=1)
    ratio=lambda a,b:a/np.maximum(b,1e-24)
    ea,eb=e[:,:,0],e[:,:,1];la,lb=low[:,0],low[:,1]
    sets=[ea[:,[0,1,3,4]],
        np.column_stack([ratio(ea[:,1],ea[:,0]),ratio(ea[:,2],ea[:,0]),ratio(ea[:,3],la),ratio(ea[:,4],la)]),
        np.column_stack([ratio(ea[:,1],ea[:,0]),ratio(ea[:,2],ea[:,0]),ratio(eb[:,1],eb[:,0]),ratio(eb[:,2],eb[:,0])]),
        np.column_stack([ratio(ea[:,2],ea[:,0]),ratio(eb[:,2],eb[:,0]),ratio(ea[:,3]+eb[:,3],la+lb),ratio(ea[:,4]+eb[:,4],la+lb)]),
        ratio(ea[:,[0,2,3,4]],eb[:,[0,2,3,4]])]
    return np.log(np.maximum(np.column_stack(sets),1e-12))

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",type=Path,required=True)
    DATASET=parser.parse_args().dataset.resolve()
    rows=list(csv.DictReader((ROOT/'config/vibration_manifest.csv').open(encoding='utf-8-sig')))
    xs={n:[] for n in [4096,8192]};ys=[];gs=[];ts=[];acq=[]
    classes=['Normal','BPFI','BPFO','Misalign','Unbalance']
    for g,r in enumerate(rows):
        s=loadmat(DATASET/r['path'],squeeze_me=True,struct_as_record=False)['Signal']
        a=s.y_values.values*float(r['conversion_factor'])+float(r['conversion_offset'])
        centers=np.arange(4096,len(a)-4096+1,4096)
        for n in xs:
            f=np.fft.rfftfreq(n,1/25600);parts=[]
            for start in range(0,len(centers),64):
                ids=centers[start:start+64,None]+np.arange(-n//2,n//2)[None,:]
                parts.append(form_features(power(a[ids]),f))
            x=np.vstack(parts);assert x.shape==(len(centers),20) and np.isfinite(x).all();xs[n].append(x)
        label=classes.index(r['class_name']);ys.append(np.full(len(centers),label));gs.append(np.full(len(centers),g));ts.append(centers/25600)
        acq.append(dict(group=g,key=r['group'],label=label,load=int(r['load_nm']),windows=len(centers),path=r['path']))
        print(g,r['group'],len(centers),flush=True)
    np.savez_compressed(OUT/'features.npz',**{f'X{n}':np.vstack(v) for n,v in xs.items()},y=np.concatenate(ys),groups=np.concatenate(gs),center_s=np.concatenate(ts))
    (OUT/'acquisitions.json').write_text(json.dumps(acq,indent=2));(OUT/'feature_sets.json').write_text(json.dumps(SETS,indent=2))
if __name__=='__main__':main()
