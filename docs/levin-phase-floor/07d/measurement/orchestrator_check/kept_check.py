import hashlib, os, re, glob, sys
os.chdir(os.path.expanduser("~/Documents/Code/SecondaryGWKit"))
def h(p):
    s=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): s.update(b)
    return s.hexdigest()
lists=sorted(set(glob.glob("var/levin-phase-floor-07*/**/KEPT_ON_MAC_STUDIO.txt",recursive=True)))
lists=[l for l in lists if "07d" not in l]
tot=ok=0; bad=[]
for L in lists:
    base=os.path.dirname(L)
    for line in open(L):
        m=re.match(r"^([0-9a-f]{64})\s+(\d+)\s+(\S.*)$",line.strip())
        if not m: continue
        sha,size,p=m.groups()
        cands=[os.path.join(base,p),p,os.path.join("var/runs",p),os.path.join(base,"..",p),os.path.join("var",p)]
        f=next((c for c in cands if os.path.exists(c)),None)
        tot+=1
        if f is None: bad.append((L,p,"missing")); continue
        if os.path.getsize(f)!=int(size) or h(f)!=sha: bad.append((L,p,"differs",os.path.getsize(f),int(size)))
        else: ok+=1
print("lists",len(lists),"entries",tot,"match",ok)
for b in bad: print(b)
# scratch copy
C="var/levin-phase-floor-06/wider/state/copy_files_certified.txt"
for line in open(C):
    m=re.search(r"([0-9a-f]{64})",line)
    if not m: continue
    names=[w for w in line.split() if w.endswith(".sqlite")]
    for n in names:
        p=os.path.join("var/levin-phase-floor-06/wider/state/a3v2_copy",os.path.basename(n))
        if os.path.exists(p): print("copy",os.path.basename(n),"MATCH" if h(p)==m.group(1) else "DIFFERS")
