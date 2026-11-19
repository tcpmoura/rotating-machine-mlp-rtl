"""Prepare a separate development candidate by mean inner scores, seed 42.

Reuses the project's consolidation rule; never ranks by external accuracy.
Does not replace the production model or claim independent final-fit accuracy.
"""
from pathlib import Path
import json,pickle,warnings,hashlib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/model_capacity/final_candidate'
threadpool_limits(1)
def main():
    r=json.loads((ROOT/'results/model_capacity/comparison.json').read_text(encoding='utf-8'))
    configs=[v['config'] for v in r['folds'][0]['selection']]
    rank=[]
    for cfg in configs:
        values=[score for fold in r['folds'] for candidate in fold['selection'] if candidate['config']==cfg for score in candidate['inner_macro_f1']]
        assert len(values)==36
        rank.append(dict(config=cfg,mean_inner_macro_f1=float(np.mean(values)),inner_scores=values))
    best=max(range(len(rank)),key=lambda i:rank[i]['mean_inner_macro_f1']);cfg=rank[best]['config']
    base=ROOT/'results/dual_housing';data=np.load(base/'features.npz');g=data['groups'];y=data['y'];X=data['X4096'][:,[8,9,10,11]]
    sample=np.concatenate([idx[np.linspace(0,len(idx)-1,min(256,len(idx)),dtype=int)] for group in np.unique(g) for idx in [np.flatnonzero(g==group)]])
    s=StandardScaler().fit(X[sample]);z=s.transform(X[sample]);counts=np.bincount(y[sample],minlength=5)
    m=MLPClassifier(hidden_layer_sizes=(cfg['hidden'],),activation='relu',solver='lbfgs',alpha=cfg['alpha'],
                    max_iter=2000,max_fun=30000,tol=1e-5,random_state=42,early_stopping=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');m.fit(z,y[sample],sample_weight=len(sample)/(5*counts[y[sample]]))
    OUT.mkdir(exist_ok=True)
    config=dict(window=4096,transform='harmonic_ab',solver='lbfgs',hidden=cfg['hidden'],alpha=cfg['alpha'],balanced=True)
    with (OUT/'model.pkl').open('wb') as f:pickle.dump(dict(model=m,scaler=s,config=config),f)
    meta=dict(status='Separate development candidate, not production; no held-out accuracy assigned.',
              selection_rule='Maximum mean of 36 inner scores; ties follow grid order, as in prior consolidation rule.',
              ranking=rank,selected_index=best,config=config,seed=42,training_acquisitions=45,training_windows=len(sample),
              convergence=dict(iterations=int(m.n_iter_),loss=float(m.loss_),warnings=[str(v.message) for v in w]),
              comparison_sha256=hashlib.sha256((ROOT/'results/model_capacity/comparison.json').read_bytes()).hexdigest())
    (OUT/'selection.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['config','training_windows','convergence']},indent=2))
if __name__=='__main__':main()
