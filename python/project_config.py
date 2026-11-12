"""Repository paths, frozen model constants and acquisition-level splits."""
from pathlib import Path
import csv,runpy
ROOT=Path(__file__).resolve().parents[1]
MODEL=runpy.run_path(str(ROOT/'model/parameters.py'))
def acquisition_split(fold):
    with (ROOT/'config/acquisition_folds.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    return dict(train_acquisitions=[r['acquisition'] for r in rows if int(r['test_fold'])!=fold],
                test_acquisitions=[r['acquisition'] for r in rows if int(r['test_fold'])==fold])
