"""Controlled nested capacity/regularization study; see README reproduction protocol."""
from pathlib import Path
import json,pickle,hashlib,time,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
from project_config import acquisition_split
OUT=ROOT/'results/model_capacity'
threadpool_limits(1)
GRID=[dict(hidden=h,alpha=a) for h in [8,16] for a in [.01,.1,1.]]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def metric(y,p):return dict(n=len(y),accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(5),average='macro',zero_division=0)),recall=recall_score(y,p,labels=range(5),average=None,zero_division=0).tolist(),confusion=confusion_matrix(y,p,labels=range(5)).tolist())
def main():
    OUT.mkdir(parents=True,exist_ok=True);base=ROOT/'results/dual_housing';d=np.load(base/'features.npz');acq=read(base/'acquisitions.json')
    X=d['X4096'][:,[8,9,10,11]];y,g=d['y'],d['groups'];labels=np.array([a['label'] for a in acq]);start=time.time()
    protocol_hash=hashlib.sha256((ROOT/'python/evaluate_model_capacity.py').read_bytes()).hexdigest()
    def sample(gids):
        return np.concatenate([ids[np.linspace(0,len(ids)-1,min(256,len(ids)),dtype=int)] for a in gids for ids in [np.flatnonzero(g==a)]])
    def fit(cfg,idx,seed):
        scaler=StandardScaler().fit(X[idx]);z=scaler.transform(X[idx]);counts=np.bincount(y[idx],minlength=5)
        weights=len(idx)/(5*counts[y[idx]])
        m=MLPClassifier(hidden_layer_sizes=(cfg['hidden'],),activation='relu',solver='lbfgs',alpha=cfg['alpha'],
                        max_iter=2000,max_fun=30000,random_state=seed,tol=1e-5,early_stopping=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');m.fit(z,y[idx],sample_weight=weights)
        return m,scaler,dict(iterations=int(m.n_iter_),loss=float(m.loss_),warnings=[str(v.message) for v in w],training=metric(y[idx],m.predict(z)))
    folds=[];summaries=[];conditions=[]
    for proto,dirname in [('acquisition','dual_housing'),('load','dual_housing_load')]:
      for seed in [17,42,73]:
        pred={p:np.full(len(y),-1) for p in ['fixed8','fixed16','inner_selected']};coverage=np.zeros(len(y),int)
        for fold in range(3):
            tag=f'{proto}_seed{seed}_fold{fold}';target=OUT/tag;target.mkdir(exist_ok=True)
            source=ROOT/'config/acquisition_folds.csv'
            meta=acquisition_split(fold) if proto=='acquisition' else dict(
                train_acquisitions=[a['key'] for a in acq if a['load']!=[0,2,4][fold]],
                test_acquisitions=[a['key'] for a in acq if a['load']==[0,2,4][fold]])
            tr=np.array([a['group'] for a in acq if a['key'] in meta['train_acquisitions']]);te=np.array([a['group'] for a in acq if a['key'] in meta['test_acquisitions']])
            assert not set(tr)&set(te) and len(tr)+len(te)==45
            test_idx=np.flatnonzero(np.isin(g,te));coverage[test_idx]+=1
            if (target/'results.json').exists():
                r=read(target/'results.json');assert r['protocol_sha256']==protocol_hash
                p=np.load(target/'predictions.npz');assert np.array_equal(p['index'],test_idx)
                for key in pred:pred[key][test_idx]=p[key]
                folds.append(r);print('Restored '+tag,flush=True);continue
            if proto=='acquisition':inner=list(StratifiedKFold(2,shuffle=True,random_state=42).split(tr,labels[tr]))
            else:
                loads=np.array([acq[k]['load'] for k in tr]);inner=[(np.flatnonzero(loads!=l),np.flatnonzero(loads==l)) for l in sorted(set(loads))]
            scores=[]
            for cfg in GRID:
                vals=[];details=[]
                for it,iv in inner:
                    ti,vi=sample(tr[it]),sample(tr[iv])
                    assert not set(g[ti])&set(g[vi]) and not set(g[ti])&set(te) and not set(g[vi])&set(te)
                    m,s,conv=fit(cfg,ti,seed);p=m.predict(s.transform(X[vi]));mm=metric(y[vi],p);vals.append(mm['macro_f1'])
                    details.append(dict(train_groups=tr[it].tolist(),validation_groups=tr[iv].tolist(),convergence=conv,validation=mm))
                scores.append(dict(config=cfg,inner_macro_f1=vals,mean=float(np.mean(vals)),details=details))
            choices={f'fixed{h}':max((i for i,c in enumerate(GRID) if c['hidden']==h),key=lambda i:scores[i]['mean']) for h in [8,16]}
            choices['inner_selected']=max(range(len(GRID)),key=lambda i:scores[i]['mean'])
            fitted={};results={}
            for policy,choice in choices.items():
                if choice not in fitted:
                    m,s,conv=fit(GRID[choice],sample(tr),seed);p=m.predict(s.transform(X[test_idx]));fitted[choice]=(p,conv)
                    with (target/f'model_{choice}.pkl').open('wb') as f:pickle.dump(dict(model=m,scaler=s,config=GRID[choice]),f)
                p,conv=fitted[choice];pred[policy][test_idx]=p
                results[policy]=dict(config=GRID[choice],choice=choice,inner_score=scores[choice]['mean'],convergence=conv,test=metric(y[test_idx],p))
            r=dict(protocol=proto,seed=seed,fold=fold,protocol_sha256=protocol_hash,source_split_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   train_acquisitions=meta['train_acquisitions'],test_acquisitions=meta['test_acquisitions'],selection=scores,policies=results)
            (target/'results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
            np.savez_compressed(target/'predictions.npz',index=test_idx,**{key:p[test_idx] for key,p in pred.items()});folds.append(r)
            print(tag,[(p,v['config'],round(v['test']['accuracy'],4)) for p,v in results.items()],f'{time.time()-start:.1f}s',flush=True)
        assert np.all(coverage==1) and all(np.all(p>=0) for p in pred.values())
        summaries.append(dict(protocol=proto,seed=seed,policies={k:metric(y,p) for k,p in pred.items()}))
        for a in acq:
            mask=g==a['group'];conditions.append(dict(protocol=proto,seed=seed,key=a['key'],label=a['label'],n=int(mask.sum()),
                    policies={k:metric(y[mask],p[mask]) for k,p in pred.items()}))
        np.savez_compressed(OUT/f'{proto}_seed{seed}_predictions.npz',y=y,groups=g,**pred)
    assert hashlib.sha256((ROOT/'python/evaluate_model_capacity.py').read_bytes()).hexdigest()==protocol_hash
    result=dict(protocol_sha256=protocol_hash,folds=folds,summaries=summaries,conditions=conditions,elapsed_seconds=time.time()-start)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('PASS: nested comparison complete.',flush=True)
if __name__=='__main__':main()
