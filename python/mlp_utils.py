import numpy as np
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
CLASSES=["Normal","BPFI","BPFO","Misalign","Unbalance"]

def export_rom(path,params):
    w1,b1,w2,b2=params
    data=np.concatenate([w1.T.ravel(),b1,w2.T.ravel(),b2])
    path.write_text('// 85 INT8 parameters; scaling specified by experiment metadata\n'+
                    '\n'.join(f'{int(v)&255:02x}' for v in data)+'\n',encoding='ascii')


def metrics(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro')),
                confusion_matrix=confusion_matrix(y,p,labels=range(5)).tolist(),
                per_class=classification_report(y,p,labels=range(5),target_names=CLASSES,output_dict=True,zero_division=0))
