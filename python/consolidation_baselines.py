"""Matched nested software controls; same acquisition/load splits as the MLP."""
from pathlib import Path
import argparse,json,itertools,pickle,warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from mlp_utils import metrics
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/dual_housing'
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['acquisition','load'],required=True);args=parser.parse_args()
OUT=ROOT/'results/consolidation'/args.split;OUT.mkdir(parents=True,exist_ok=True)
d=np.load(BASE/'features.npz');X={n:d[f'X{n}'] for n in [4096,8192]};y,groups=d['y'],d['groups']
acq=json.loads((BASE/'acquisitions.json').read_text());labels=np.array([a['label'] for a in acq])
sets=json.loads((BASE/'feature_sets.json').read_text())
grid=[dict(window=n,features=f,C=c) for n,f,c in itertools.product([4096,8192],sets,[1,10])]
def sample(gids):
    return np.concatenate([idx[np.linspace(0,len(idx)-1,min(256,len(idx)),dtype=int)] for g in gids for idx in [np.flatnonzero(groups==g)]])
def fit(kind,cfg,idx):
    est=LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=2000,random_state=42) if kind=='logistic' else SVC(C=cfg['C'],gamma='scale',class_weight='balanced')
    m=make_pipeline(StandardScaler(),est)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');m.fit(X[cfg['window']][idx][:,sets[cfg['features']]],y[idx])
    return m,[str(v.message) for v in w]
def predict(m,cfg,idx):return m.predict(X[cfg['window']][idx][:,sets[cfg['features']]])
def main():
  for kind in ['logistic','rbf_svm']:
    out=OUT/kind;out.mkdir(exist_ok=True);pred=np.full(len(y),-1);coverage=np.zeros(len(y),int);results=[]
    for f in range(3):
      previous=ROOT/'results'/('dual_housing' if args.split=='acquisition' else 'dual_housing_load')/f'fold_{f}/metrics.json'
      meta=json.loads(previous.read_text());tr=np.array([a['group'] for a in acq if a['key'] in meta['train_acquisitions']]);te=np.array([a['group'] for a in acq if a['key'] in meta['test_acquisitions']])
      assert not set(tr)&set(te);vi=np.flatnonzero(np.isin(groups,te))
      final=out/f'fold_{f}.json'
      if final.exists():
        result=json.loads(final.read_text());pred[vi]=np.load(out/f'fold_{f}_pred.npy');coverage[vi]+=1;results.append(result);continue
      if args.split=='acquisition':inner=list(StratifiedKFold(2,shuffle=True,random_state=42).split(tr,labels[tr]))
      else:
        loads=np.array([acq[g]['load'] for g in tr]);inner=[(np.flatnonzero(loads!=l),np.flatnonzero(loads==l)) for l in sorted(set(loads))]
      checkpoint=out/f'inner_{f}.json';scores=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
      for ci,cfg in enumerate(grid):
        if ci<len(scores):assert scores[ci]['config']==cfg;continue
        vals=[];warn=[]
        for it,iv in inner:
          ti,valid=sample(tr[it]),sample(tr[iv]);m,w=fit(kind,cfg,ti);warn.extend(w)
          vals.append(float(f1_score(y[valid],predict(m,cfg,valid),average='macro',zero_division=0)))
        scores.append(dict(config=cfg,inner_macro_f1=vals,mean=float(np.mean(vals)),warnings=warn))
        tmp=checkpoint.with_suffix('.tmp');tmp.write_text(json.dumps(scores,indent=2));tmp.replace(checkpoint)
      chosen=max(scores,key=lambda s:s['mean'])['config'];m,w=fit(kind,chosen,sample(tr));p=predict(m,chosen,vi)
      result=dict(fold=f,split=args.split,model=kind,config=chosen,selection=scores,warnings=w,
                  train_acquisitions=meta['train_acquisitions'],test_acquisitions=meta['test_acquisitions'],metrics=metrics(y[vi],p))
      np.save(out/f'fold_{f}_pred.npy',p)
      with (out/f'fold_{f}_model.pkl').open('wb') as file:pickle.dump(m,file)
      final.write_text(json.dumps(result,indent=2));pred[vi]=p;coverage[vi]+=1;results.append(result)
      print(args.split,kind,f,chosen,result['metrics']['accuracy'],flush=True)
    assert np.all(coverage==1) and np.all(pred>=0)
    summary=dict(model=kind,split=args.split,folds=results,pooled=metrics(y,pred),mean_acquisition_accuracy=float(np.mean([np.mean(pred[groups==g]==y[groups==g]) for g in range(45)])))
    (out/'results.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'predictions.npz',y=y,predictions=pred,groups=groups)
    print('POOLED',args.split,kind,summary['pooled']['accuracy'],flush=True)
if __name__=='__main__':main()
