import json, sys
D, A, row = sys.argv[1], sys.argv[2], sys.argv[3]
def get(p, unit):
    for l in open(p):
        r = json.loads(l)
        if r['unit'] == unit: return r['data']
a = get(f"{D}/checkpoint.jsonl", f"t5-1e-7-{row}"); b = get(f"{A}/checkpoint.jsonl", f"t-1e-7-{row}")
print('row', row, 'role', a['role'], b.get('role'), 'tree_head', a['tree_head'][:7], 'vs 05a', b.get('tree_head','')[:7])
ok = True
for k in ('total', 'total_abserr'):
    x, y = a['result'][k]['hex'], b['result'][k]['hex']; ok &= x == y; print(k, x, y, x == y)
print('calls', len(a['calls']), len(b['calls']), 'call keys', sorted(a['calls'][0].keys()))
n = 0
for i, (ca, cb) in enumerate(zip(a['calls'], b['calls'])):
    for k in ('value', 'abserr'):
        x, y = ca.get(k), cb.get(k)
        hx = x['hex'] if isinstance(x, dict) else x; hy = y['hex'] if isinstance(y, dict) else y
        if hx != hy: ok = False; print('call', i, k, hx, hy)
        n += 1
print('call fields compared', n, 'all equal' if ok and len(a['calls']) == len(b['calls']) else 'DIFFER')
