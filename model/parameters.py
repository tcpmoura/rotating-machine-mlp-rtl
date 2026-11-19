"""Frozen INT16 development model; no independent final-fit accuracy."""
CLASSES = ['Normal', 'BPFI', 'BPFO', 'Misalign', 'Unbalance']
CONFIG = {'alpha': 1.0,
 'balanced': True,
 'hidden': 8,
 'solver': 'lbfgs',
 'transform': 'harmonic_ab',
 'window': 4096}
FEATURE_COLUMNS = [8, 9, 10, 11]
MEAN = [0.12764074718924912,
 -0.7405183063639997,
 1.2329157579686745,
 1.1073230840095425]
SCALE = [0.8572691039471241, 2.292448701113573, 0.7030957378277766, 1.2983469305941027]
LOG_FLOOR = 1e-12
SCALES = {'hidden_frac': 11, 'input_frac': 13, 'w1_frac': 12, 'w2_frac': 12}
FIT_SEED = 42
DATA_W = 16
ACC_W = 36
