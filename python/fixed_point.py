"""Bit-exact model of the 4-8-5 RTL, independent of scikit-learn.

Fractional scales are supplied by the caller; model/parameters.py records
the released model's scales. Hidden rounding is half-up for nonnegative values.
"""
from pathlib import Path
import numpy as np


def read_rom(path):
    values=[]
    # Original ROM comments use Windows-1252; hexadecimal payload is ASCII.
    for line in Path(path).read_text(encoding='cp1252').splitlines():
        for token in line.split('//')[0].split():
            v=int(token,16)
            values.append(v-256 if v>127 else v)
    if len(values)!=85: raise ValueError(f'Expected 85 ROM bytes, got {len(values)}')
    r=np.array(values,dtype=np.int64)
    return r[:32].reshape(8,4).T,r[32:40],r[40:80].reshape(5,8).T,r[80:85]


def infer(x,params,scales=None):
    w1,b1,w2,b2=params
    scales=scales or dict(input_frac=5,w1_frac=5,w2_frac=5,hidden_frac=5)
    input_scale=1<<scales['input_frac'];hidden_scale=1<<scales['hidden_frac']
    hidden_shift=scales['input_frac']+scales['w1_frac']-scales['hidden_frac']
    score_shift=scales['hidden_frac']+scales['w2_frac']-5
    x=np.asarray(x,dtype=np.int64)
    a1=x@w1+b1*input_scale
    if hidden_shift<1:raise ValueError('Hidden shift must be positive')
    hidden=np.minimum((np.maximum(a1,0)+(1<<(hidden_shift-1)))>>hidden_shift,127)
    output=hidden@w2+b2*hidden_scale
    cls=output.argmax(axis=-1)
    score=np.take_along_axis(output,cls[...,None],axis=-1)[...,0]
    score=score>>score_shift if score_shift>=0 else score<<(-score_shift)
    # Revised interface saturates BOTH ends; original saturated only +127.
    confidence=np.clip(score,-128,127)
    return hidden,output,cls,confidence


def write_vectors(path,x,params,scales=None):
    h,o,c,s=infer(x,params,scales)
    with Path(path).open('w',encoding='ascii') as f:
        for row,hr,orr,cr,sr in zip(x,h,o,c,s):
            tokens=[*(f'{int(v)&255:02x}' for v in row),
                    *(f'{int(v)&255:02x}' for v in hr),
                    *(f'{int(v)&0xffffff:06x}' for v in orr),
                    f'{int(cr):02x}',f'{int(sr)&255:02x}']
            f.write(' '.join(tokens)+'\n')


if __name__=='__main__':
    import argparse,itertools
    p=argparse.ArgumentParser();p.add_argument('--rom',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--random',type=int,default=10000)
    args=p.parse_args()
    edge=np.array(list(itertools.product([-128,-1,0,1,127],repeat=4)))
    random=np.random.default_rng(42).integers(-128,128,(args.random,4))
    x=np.vstack((edge,random));write_vectors(args.output,x,read_rom(args.rom))
    print(f'Exported {len(x)} deterministic arithmetic stress vectors (not diagnostic accuracy).')
