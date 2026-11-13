import sqlite3,json,sys
from pathlib import Path
p=Path(sys.argv[1])
con=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
for meta,tc,md in con.execute('select metadata,total_converged,WKB_Levin_max_depth from QuadSourceIntegral'):
    m=json.loads(meta)
    if m.get('WKB_Levin') and m.get('analytic'):
        break
def sk(o,d=0):
  if isinstance(o,dict):
    for k,v in o.items():
      if isinstance(v,(dict,list)): print(' '*d+k); sk(v,d+2)
      else: print(' '*d+k, repr(v)[:60])
  elif isinstance(o,list):
    print(' '*d+'[list len %d]'%len(o))
    if o: sk(o[0],d+2)
sk(m); print(tc,md)
