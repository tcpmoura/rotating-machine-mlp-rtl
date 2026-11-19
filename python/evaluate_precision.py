"""New fixed8 models: train-only numeric calibration and generic vector export."""
from pathlib import Path
import json,pickle,hashlib,itertools
import numpy as np
from numeric_formats import calibrate,parameters,evaluate,metrics,quant,validate_arithmetic
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/candidate_precision'

def configure(m,z,width):
    h=np.maximum(z@m.coefs_[0]+m.intercepts_[0],0)
    return dict(input_frac=calibrate([z],width),w1_frac=calibrate([m.coefs_[0],m.intercepts_[0]],width),
                w2_frac=calibrate([m.coefs_[1],m.intercepts_[1]],width),hidden_frac=calibrate([h],width))

def export(folder,z,m,sc,width,stress=False):
    folder.mkdir(parents=True,exist_ok=True);acc=max(24,2*width+4);pars=parameters(m,sc,(width,)*4)
    p,o,h,diag=evaluate(z,pars,sc,(width,)*4);assert diag['conservative_accumulator_bits']<=acc
    x=quant(z,sc['input_frac'],width);cs=sc['hidden_frac']+sc['w2_frac']-5
    best=o[np.arange(len(p)),p];score=np.clip(best>>cs if cs>=0 else best<<(-cs),-128,127)
    payload=np.concatenate([pars[0].T.ravel(),pars[1],pars[2].T.ravel(),pars[3]])
    (folder/'weights_rom.mem').write_text('\n'.join(f'{int(v)&((1<<width)-1):0{(width+3)//4}x}' for v in payload)+'\n')
    with (folder/'vectors.txt').open('w') as f:
        for xx,hh,oo,pp,ss in zip(x,h,o,p,score):
            f.write(' '.join([*(f'{int(v)&((1<<width)-1):x}' for v in xx),*(f'{int(v)&((1<<width)-1):x}' for v in hh),
                              *(f'{int(v)&((1<<acc)-1):x}' for v in oo),f'{int(pp):x}',f'{int(ss)&255:x}'])+'\n')
    meta=dict(width=width,acc_width=acc,scales=sc,n=len(z),diagnostics=diag,stress=stress)
    (folder/'parameters.json').write_text(json.dumps(meta,indent=2));return p,meta

def main():
    validate_arithmetic();OUT.mkdir(parents=True,exist_ok=True)
    data=np.load(ROOT/'results/dual_housing/features.npz');acq=json.loads((ROOT/'results/dual_housing/acquisitions.json').read_text());X=data['X4096'][:,[8,9,10,11]];y,g=data['y'],data['groups']
    source=ROOT/'results/model_capacity';results=json.loads((source/'comparison.json').read_text());folds=[];summaries=[];conditions=[];hashes={}
    for protocol in ['acquisition','load']:
      for seed in [17,42,73]:
        pred={k:np.full(len(y),-1) for k in ['float','int8','int12','int16']};coverage=np.zeros(len(y),int)
        for r in results['folds']:
            if r['protocol']!=protocol or r['seed']!=seed:continue
            tag=f"{protocol}_seed{seed}_fold{r['fold']}";src=source/tag
            model_path=src/f"model_{r['policies']['fixed8']['choice']}.pkl"
            with model_path.open('rb') as f:b=pickle.load(f)
            hashes[str(model_path.relative_to(ROOT))]=hashlib.sha256(model_path.read_bytes()).hexdigest()
            m,s=b['model'],b['scaler'];assert m.hidden_layer_sizes==(8,)
            tr=[a['group'] for a in acq if a['key'] in r['train_acquisitions']];te=[a['group'] for a in acq if a['key'] in r['test_acquisitions']]
            assert not set(tr)&set(te)
            ti=np.flatnonzero(np.isin(g,tr));vi=np.flatnonzero(np.isin(g,te));coverage[vi]+=1
            ztr=s.transform(X[ti]);z=s.transform(X[vi]);pf=m.predict(z);saved=np.load(src/'predictions.npz')
            assert np.array_equal(vi,saved['index']) and np.array_equal(pf,saved['fixed8']);pred['float'][vi]=pf
            row=dict(protocol=protocol,seed=seed,fold=r['fold'],float=metrics(y[vi],pf,pf),formats={})
            for width in [8,12,16]:
                sc=configure(m,ztr,width);p,meta=export(OUT/'folds'/tag/f'int{width}',z,m,sc,width)
                pred[f'int{width}'][vi]=p;row['formats'][f'int{width}']=dict(**meta,metrics=metrics(y[vi],p,pf))
            folds.append(row)
        assert np.all(coverage==1)
        summaries.append(dict(protocol=protocol,seed=seed,formats={k:metrics(y,p,pred['float']) for k,p in pred.items()}))
        for a in acq:
            mask=g==a['group'];conditions.append(dict(protocol=protocol,seed=seed,key=a['key'],label=a['label'],formats={k:metrics(y[mask],p[mask],pred['float'][mask]) for k,p in pred.items()}))
        np.savez_compressed(OUT/f'{protocol}_seed{seed}_predictions.npz',y=y,groups=g,**pred)
    final_path=source/'final_candidate/model.pkl'
    with final_path.open('rb') as f:b=pickle.load(f)
    hashes[str(final_path.relative_to(ROOT))]=hashlib.sha256(final_path.read_bytes()).hexdigest()
    final=[];z=b['scaler'].transform(X)
    for width in [8,12,16]:
        sc=configure(b['model'],z,width);_,meta=export(OUT/'final'/f'int{width}',z,b['model'],sc,width);final.append(meta)
        edge=[-2**(width-1),-1,0,1,2**(width-1)-1]
        x=np.vstack([np.array(list(itertools.product(edge,repeat=4))),np.random.default_rng(42).integers(-2**(width-1),2**(width-1),(1000,4))])
        export(OUT/'stress'/f'int{width}',x/2.**sc['input_frac'],b['model'],sc,width,True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'comparison.json').write_text(json.dumps(dict(source_sha256=hashes,folds=folds,summaries=summaries,conditions=conditions,final=final),indent=2))
    print('PASS: 18 new fixed8 models and development candidate; 3 widths, train-only fold calibration.')
    print(json.dumps(final,indent=2))
if __name__=='__main__':main()
