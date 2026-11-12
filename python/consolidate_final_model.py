"""Final development fit selected from internal scores only; no independent test claim."""
from pathlib import Path
import json,pickle,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from mlp_utils import metrics,export_rom
from fixed_point import infer,write_vectors
from calibrate import q,fraction_bits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/dual_housing'
OUT=ROOT/'results/final_model';OUT.mkdir(parents=True,exist_ok=True)
def main():
    selections=[];sources=[]
    for protocol in ['dual_housing','dual_housing_load']:
        for seed in [17,42,73]:
            folder=protocol+(f'_seed{seed}' if seed!=42 else '')
            r=json.loads((ROOT/'results'/folder/'results.json').read_text())
            for f in r['folds']:selections.append(f['selection'])
            sources.append(folder)
    first=selections[0];rank=[]
    for i,record in enumerate(first):
        cfg=record['config'];values=[]
        for selection in selections:
            assert selection[i]['config']==cfg
            values+=selection[i]['inner_macro_f1']
        rank.append(dict(config=cfg,internal_macro_f1=values,mean=float(np.mean(values))))
    chosen=max(range(len(rank)),key=lambda i:rank[i]['mean']);cfg=rank[chosen]['config']
    selection=dict(rule='Mean internal macro-F1 across three seeds and both protocols, equal weight per internal fold',sources=sources,selected_index=chosen,config=cfg,candidates=rank,fit_seed=42,independent_test=False)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    d=np.load(BASE/'features.npz');sets=json.loads((BASE/'feature_sets.json').read_text());X=d[f"X{cfg['window']}"][:,sets[cfg['transform']]];y,g=d['y'],d['groups']
    idx=np.concatenate([ix[np.linspace(0,len(ix)-1,min(256,len(ix)),dtype=int)] for k in np.unique(g) for ix in [np.flatnonzero(g==k)]])
    scaler=StandardScaler().fit(X[idx]);z=scaler.transform(X);counts=np.bincount(y[idx],minlength=5);weights=len(idx)/(5*counts[y[idx]])
    model=MLPClassifier(hidden_layer_sizes=(8,),activation='relu',solver=cfg['solver'],alpha=cfg['alpha'],
        batch_size=256,max_iter=2000,max_fun=30000,random_state=42,early_stopping=False,n_iter_no_change=40,tol=1e-5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');model.fit(z[idx],y[idx],sample_weight=weights)
    hidden=np.maximum(z@model.coefs_[0]+model.intercepts_[0],0)
    scales=dict(input_frac=fraction_bits([z]),w1_frac=fraction_bits([model.coefs_[0],model.intercepts_[0]]),
                w2_frac=fraction_bits([model.coefs_[1],model.intercepts_[1]]),hidden_frac=fraction_bits([hidden]))
    assert min(scales.values())>=0
    params=(q(model.coefs_[0],scales['w1_frac']),q(model.intercepts_[0],scales['w1_frac']),q(model.coefs_[1],scales['w2_frac']),q(model.intercepts_[1],scales['w2_frac']))
    xq=q(z,scales['input_frac']);_,_,pi,_=infer(xq,params,scales)
    export_rom(OUT/'weights_rom.mem',params);write_vectors(OUT/'vectors.txt',xq,params,scales)
    power_idx=np.array([ix[len(ix)//2] for c in range(5) for ix in [np.flatnonzero(y==c)]])
    write_vectors(OUT/'power_vectors.txt',xq[power_idx],params,scales)
    (OUT/'scales.json').write_text(json.dumps(scales,indent=2))
    with (OUT/'model.pkl').open('wb') as f:pickle.dump(dict(model=model,scaler=scaler,config=cfg,scales=scales),f)
    (OUT/'preprocessing.json').write_text(json.dumps(dict(config=cfg,columns=sets[cfg['transform']],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),log_floor=1e-12,input_frac=scales['input_frac']),indent=2))
    result=dict(config=cfg,scales=scales,fit_seed=42,training_windows=len(idx),development_windows=len(y),iterations=model.n_iter_,loss=float(model.loss_),warnings=[str(v.message) for v in w],
                development_float=metrics(y,model.predict(z)),development_int8=metrics(y,pi),independent_test=False,power_indices=power_idx.tolist())
    (OUT/'development_fit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if not k.startswith('development_')},indent=2))
if __name__=='__main__':main()
