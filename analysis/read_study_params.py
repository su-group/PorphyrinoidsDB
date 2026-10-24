"""Print the best params of every model study in a run directory.

Usage: python read_study_params.py <models/run dir> [model,model,...]
"""
import sys
import sqlite3
from pathlib import Path

run_dir = Path(sys.argv[1])
only = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None

for db in sorted(run_dir.glob('*.sqlite3')):
    model = db.stem
    if only and model not in only:
        continue
    con = sqlite3.connect('file:%s?mode=ro' % db.as_posix(), uri=True)
    cur = con.cursor()
    cur.execute('SELECT trial_id, value FROM trial_values ORDER BY value ASC LIMIT 1')
    row = cur.fetchone()
    if row is None:
        print('%-16s (no completed trial)' % model)
        con.close()
        continue
    trial_id, value = row
    cur.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id=?', (trial_id,))
    params = {n: float(v) for n, v in cur.fetchall()}
    pretty = ', '.join('%s=%s' % (k, ('%d' % v) if float(v).is_integer() else ('%.8g' % v))
                       for k, v in sorted(params.items()))
    cur.execute('SELECT COUNT(*), SUM(state=1) FROM trials')
    total, done = cur.fetchone()
    print('%-16s trials=%d scored=%d  CV MSE=%.12f  %s' % (model, total, done or 0, value, pretty))
    con.close()
