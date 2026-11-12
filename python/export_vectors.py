"""Regenerate all final-model integer vectors from extracted development features."""
from pathlib import Path
import argparse,json
import numpy as np
from fixed_point import read_rom,write_vectors
from calibrate import q
from project_config import MODEL
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=ROOT/'results/dual_housing/features.npz');p.add_argument('--output',type=Path,default=ROOT/'build/full_vectors.txt');args=p.parse_args()
    meta=dict(config=MODEL['CONFIG'],columns=MODEL['FEATURE_COLUMNS'],mean=MODEL['MEAN'],scale=MODEL['SCALE']);scales=MODEL['SCALES']
    d=np.load(args.features);x=d[f"X{meta['config']['window']}"][:,meta['columns']]
    z=(x-np.array(meta['mean']))/np.array(meta['scale']);args.output.parent.mkdir(parents=True,exist_ok=True)
    write_vectors(args.output,q(z,scales['input_frac']),read_rom(ROOT/'model/weights_rom.mem'),scales)
    print(f'{len(z)} vectors written to {args.output}')
if __name__=='__main__':main()
