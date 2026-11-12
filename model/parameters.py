"""Frozen final-model constants. Feature order is the hardware input order.

4096-sample Hann windows; log harmonic ratios E2/E1 and E3/E1 at housings A/B.
Standardize with MEAN/SCALE, then round and saturate to signed INT8 using SCALES.
The model was fitted with all development acquisitions, not an independent test.
"""
CLASSES = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
CONFIG = {'window': 4096,
 'transform': 'harmonic_ab',
 'solver': 'lbfgs',
 'alpha': 0.01,
 'balanced': True}
FEATURE_COLUMNS = [8, 9, 10, 11]
MEAN = [0.12764074718924912,
 -0.7405183063639997,
 1.2329157579686745,
 1.1073230840095425]
SCALE = [0.8572691039471241, 2.292448701113573, 0.7030957378277766, 1.2983469305941027]
LOG_FLOOR = 1e-12
SCALES = {'input_frac': 5, 'w1_frac': 3, 'w2_frac': 3, 'hidden_frac': 1}
FIT_SEED = 42
