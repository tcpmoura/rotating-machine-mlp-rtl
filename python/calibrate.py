import numpy as np

def fraction_bits(arrays,max_bits=5):
    bound=max(float(np.max(np.abs(a))) for a in arrays)
    return min(max_bits,int(np.floor(np.log2(127/bound)))) if bound else max_bits


def q(a,bits):return np.clip(np.rint(a*(2.0**bits)),-128,127).astype(np.int64)
