import glob
import sqlite3
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'experiments/20261014_R-MF/models/R-MF'
WANT = sys.argv[2].split(',') if len(sys.argv) > 2 else None


def cols(con, t):
    return [r[1] for r in con.execute('pragma table_info(%s)' % t)]


for db in sorted(glob.glob(ROOT + '/*.sqlite3')):
    name = db.replace('\\', '/').split('/')[-1].replace('.sqlite3', '')
    if WANT and name not in WANT:
        continue
    con = sqlite3.connect(db)
    try:
        tc = cols(con, 'trials')
        t = con.execute('select * from trials').fetchall()
        vc = cols(con, 'trial_values')
        v = con.execute('select * from trial_values').fetchall()
        pc = cols(con, 'trial_params')
        p = con.execute('select * from trial_params').fetchall()
    except sqlite3.Error as e:
        print('%s: DB ERROR %s' % (name, e))
        con.close()
        continue
    tix, six = tc.index('trial_id'), tc.index('state')
    vix = vc.index('trial_id')
    vm = {}
    for r in v:
        vm.setdefault(r[vix], []).append(r[vc.index('value')])
    pm = {}
    for r in p:
        pm.setdefault(r[vc.index('trial_id')] if False else r[pc.index('trial_id')], {})[
            r[pc.index('param_name')]] = r[pc.index('param_value')]
    scored = [(vm[r[tix]][0], r[tix], r[six]) for r in t if vm.get(r[tix])]
    scored.sort()
    print('%-16s trials=%-4d scored=%-4d' % (name, len(t), len(scored)), end='')
    if scored:
        mse, tid, st = scored[0]
        print('  best=%.4f trial=%s state=%s %s' % (mse, tid, st, pm.get(tid, {})))
    else:
        print()
    con.close()
