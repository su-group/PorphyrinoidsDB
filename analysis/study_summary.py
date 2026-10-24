"""Report the study-DB best value (the pipeline's own CV MSE) for every model of a run.

Usage: python -B analysis/study_summary.py <models_dir> [...]
Prints one line per model: name  best_value  n_complete  best_params
"""
import json
import sqlite3
import sys
from pathlib import Path

for models_dir in sys.argv[1:]:
    root = Path(models_dir)
    print(f'=== {root} ===')
    for db in sorted(root.glob('*.sqlite3')):
        name = db.stem
        try:
            con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
            cur = con.cursor()
            cur.execute("select count(*), sum(state='COMPLETE') from trials")
            total, done = cur.fetchone()
            cur.execute('select trial_id from trials where state=\'COMPLETE\' order by number')
            ids = [r[0] for r in cur.fetchall()]
            best_val, best_params = None, None
            for tid in ids:
                cur.execute('select value from trial_values where trial_id=? order by trial_id limit 1', (tid,))
                row = cur.fetchone()
                if row is None or row[0] is None:
                    continue
                if best_val is None or row[0] < best_val:
                    best_val = row[0]
            if ids:
                cur.execute('select param_name, param_value from trial_params where trial_id=?', (ids[-1],))
                best_params = dict(cur.fetchall())
            con.close()
        except Exception as exc:  # noqa: BLE001
            print(f'  {name}: ERROR {type(exc).__name__}: {exc}')
            continue
        print(f'  {name:16s} best_value={best_val!r}  trials={total} complete={done}')
        if best_params:
            pretty = json.dumps(best_params, ensure_ascii=True, default=str)
            print(f'      last_trial_params={pretty[:220]}')
    print()
