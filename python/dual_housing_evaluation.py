"""Predeclared training-only model selection; configuration selection by inner macro-F1."""
from pathlib import Path
import json,pickle,itertools,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from mlp_utils import metrics,export_rom
from fixed_point import infer,write_vectors,read_rom
from calibrate import fraction_bits,q

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'/'dual_housing'
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['acquisition','load'],default='acquisition')
parser.add_argument('--resume',action='store_true');parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
from project_config import acquisition_split
out_name=('dual_housing' if args.split=='acquisition' else 'dual_housing_load')
if args.seed!=42:out_name+=f'_seed{args.seed}'
OUT=ROOT/'results'/out_name;OUT.mkdir(parents=True,exist_ok=True)
SETS=json.loads((BASE/'feature_sets.json').read_text())
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
data=np.load(BASE/'features.npz');X,y,groups={n:data[f'X{n}'] for n in [4096,8192]},data['y'],data['groups']
acq=json.loads((BASE/'acquisitions.json').read_text())
labels=np.array([a['label'] for a in acq])
candidates=[dict(window=n,transform=t,solver=solver,alpha=.01,balanced=True) for n,t,solver in itertools.product([4096,8192],list(SETS),['adam','lbfgs'])]


def transform(x,name):
    return x[:,SETS[name]]


def sample_groups(gids):
    result=[]
    for g in gids:
        idx=np.flatnonzero(groups==g)
        result.extend(idx[np.linspace(0,len(idx)-1,min(256,len(idx)),dtype=int)])
    return np.array(result)


def fit(idx,cfg):
    scaler=StandardScaler().fit(transform(X[cfg['window']][idx],cfg['transform']))
    z=scaler.transform(transform(X[cfg['window']][idx],cfg['transform']))
    model=MLPClassifier(hidden_layer_sizes=(8,),activation='relu',solver=cfg['solver'],
        alpha=cfg['alpha'],batch_size=256,max_iter=2000,max_fun=30000,random_state=args.seed,
        early_stopping=False,n_iter_no_change=40,tol=1e-5)
    weights=None
    if cfg['balanced']:
        counts=np.bincount(y[idx],minlength=5)
        weights=len(idx)/(5*counts[y[idx]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');model.fit(z,y[idx],sample_weight=weights)
    return model,scaler,dict(warnings=[str(v.message) for v in w],iterations=model.n_iter_,loss=float(model.loss_),loss_curve=getattr(model,'loss_curve_',[]))


def main():
    pred_float=np.full(len(y),-1);pred_int=np.full(len(y),-1);results=[];covered=np.zeros(len(y),dtype=int)
    for fold in range(3):
        completed=OUT/f'fold_{fold}'
        if args.resume and (completed/'metrics.json').exists():
            result=json.loads((completed/'metrics.json').read_text())
            assert result['split']==args.split and result.get('seed',42)==args.seed
            with (completed/'model.pkl').open('rb') as f:saved=pickle.load(f)
            cfg=saved['config']
            test_g=[a['group'] for a in acq if a['key'] in result['test_acquisitions']]
            ix=np.flatnonzero(np.isin(groups,test_g))
            z=saved['scaler'].transform(transform(X[cfg['window']][ix],cfg['transform']))
            pf=saved['model'].predict(z)
            scales=result['scales'];params=read_rom(completed/'weights_rom.mem')
            _,_,pi,_=infer(q(z,scales['input_frac']),params,scales)
            assert abs(metrics(y[ix],pi)['accuracy']-result['int8']['accuracy'])<1e-12
            pred_float[ix]=pf;pred_int[ix]=pi;covered[ix]+=1;results.append(result)
            print(f'Restored {args.split} fold {fold}',flush=True)
            continue
        if args.split=='acquisition':
            meta=acquisition_split(fold)
            train_g=np.array([a['group'] for a in acq if a['key'] in meta['train_acquisitions']])
            test_g=np.array([a['group'] for a in acq if a['key'] in meta['test_acquisitions']])
            inner=list(StratifiedKFold(n_splits=2,shuffle=True,random_state=42).split(train_g,labels[train_g]))
        else:
            held_load=[0,2,4][fold]
            train_g=np.array([a['group'] for a in acq if a['load']!=held_load])
            test_g=np.array([a['group'] for a in acq if a['load']==held_load])
            train_loads=np.array([acq[g]['load'] for g in train_g])
            inner=[(np.flatnonzero(train_loads!=l),np.flatnonzero(train_loads==l)) for l in sorted(set(train_loads))]
            meta=dict(train_acquisitions=[acq[g]['key'] for g in train_g],test_acquisitions=[acq[g]['key'] for g in test_g])
        assert len(train_g)+len(test_g)==45 and not set(train_g)&set(test_g)
        assert set(labels[train_g])==set(range(5)) and set(labels[test_g])==set(range(5))
        checkpoint=OUT/f'inner_fold_{fold}.json'
        scores=json.loads(checkpoint.read_text()) if args.resume and checkpoint.exists() else []
        for ci,cfg in enumerate(candidates):
            if ci<len(scores):
                assert scores[ci]['config']==cfg
                continue
            fold_scores=[];convergence=[]
            for it,iv in inner:
                train_idx=sample_groups(train_g[it]);val_idx=sample_groups(train_g[iv])
                assert not set(groups[train_idx])&set(groups[val_idx])
                assert not set(groups[train_idx])&set(test_g)
                m,s,conv=fit(train_idx,cfg);convergence.append(conv)
                p=m.predict(s.transform(transform(X[cfg['window']][val_idx],cfg['transform'])))
                fold_scores.append(float(f1_score(y[val_idx],p,average='macro',zero_division=0)))
            score=dict(config=cfg,convergence=convergence,inner_macro_f1=fold_scores,mean=float(np.mean(fold_scores)))
            scores.append(score)
            temporary=checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(scores,indent=2));temporary.replace(checkpoint)
            print(f'{args.split} outer {fold} candidate {ci}: inner macro-F1={score["mean"]:.4f}',flush=True)
        selected=max(range(len(scores)),key=lambda k:scores[k]['mean'])
        cfg=candidates[selected];train_idx=sample_groups(train_g);test_idx=np.flatnonzero(np.isin(groups,test_g))
        model,scaler,warning=fit(train_idx,cfg)
        ztr=scaler.transform(transform(X[cfg['window']][train_idx],cfg['transform']))
        zte=scaler.transform(transform(X[cfg['window']][test_idx],cfg['transform']))
        pf=model.predict(zte)
        # Calibration can use ALL training-group features without using test or labels.
        alltrain=np.flatnonzero(np.isin(groups,train_g))
        zcal=scaler.transform(transform(X[cfg['window']][alltrain],cfg['transform']))
        hidden=np.maximum(zcal@model.coefs_[0]+model.intercepts_[0],0)
        scales=dict(input_frac=fraction_bits([zcal]),
                    w1_frac=fraction_bits([model.coefs_[0],model.intercepts_[0]]),
                    w2_frac=fraction_bits([model.coefs_[1],model.intercepts_[1]]),
                    hidden_frac=fraction_bits([hidden]))
        assert min(scales.values())>=0
        params=(q(model.coefs_[0],scales['w1_frac']),q(model.intercepts_[0],scales['w1_frac']),
                q(model.coefs_[1],scales['w2_frac']),q(model.intercepts_[1],scales['w2_frac']))
        xq=q(zte,scales['input_frac']);_,_,pi,_=infer(xq,params,scales)
        pred_float[test_idx]=pf;pred_int[test_idx]=pi
        covered[test_idx]+=1
        out=OUT/f'fold_{fold}';out.mkdir(exist_ok=True)
        export_rom(out/'weights_rom.mem',params);write_vectors(out/'vectors.txt',xq,params,scales)
        (out/'scales.json').write_text(json.dumps(scales,indent=2))
        with (out/'model.pkl').open('wb') as f:pickle.dump(dict(model=model,scaler=scaler,config=cfg),f)
        result=dict(fold=fold,split=args.split,seed=args.seed,selection=scores,selected=selected,config=cfg,scales=scales,
                    train_acquisitions=meta['train_acquisitions'],test_acquisitions=meta['test_acquisitions'],
                    training_windows=len(train_idx),test_windows=len(test_idx),iterations=model.n_iter_,convergence=warning,
                    float=metrics(y[test_idx],pf),int8=metrics(y[test_idx],pi))
        results.append(result);(out/'metrics.json').write_text(json.dumps(result,indent=2))
        print(f'OUTER {fold}: selected {cfg}, float={result["float"]["accuracy"]:.4f}, INT8={result["int8"]["accuracy"]:.4f}',flush=True)
    assert np.all(pred_float>=0) and np.all(pred_int>=0) and np.all(covered==1)
    per_acquisition=[dict(group=int(g),key=acq[g]['key'],float_accuracy=float(np.mean(pred_float[groups==g]==y[groups==g])),int8_accuracy=float(np.mean(pred_int[groups==g]==y[groups==g]))) for g in np.unique(groups)]
    (OUT/'per_acquisition.json').write_text(json.dumps(per_acquisition,indent=2))
    summary=dict(split=args.split,seed=args.seed,protocol='Dual-housing nested CV; exploratory dataset reuse',
                 folds=results,pooled_float=metrics(y,pred_float),pooled_int8=metrics(y,pred_int))
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(OUT/'predictions.npz',y=y,groups=groups,pred_float=pred_float,pred_int8=pred_int)
    print('POOLED:',summary['pooled_float']['accuracy'],summary['pooled_int8']['accuracy'],flush=True)


if __name__=='__main__':main()
