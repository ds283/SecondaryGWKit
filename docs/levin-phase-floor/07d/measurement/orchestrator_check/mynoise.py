# Re-measure sigma on two knot-free leaves by a method independent of the agent's least-squares fits:
# the n-th forward difference of samples on a uniform grid, sigma = rms(Delta^n y)/sqrt(C(2n,n)).
import json, os, sys, glob, hashlib, math
import numpy as np
M=os.path.expanduser("~/Documents/Code/SecondaryGWKit"); R=M+"/var/levin-phase-floor-07d/measurement"
sys.path.insert(1,R+"/scripts")
import AdaptiveLevin.levin_quadrature as LQ
import ray.cloudpickle as cp
from mpmath import mp; mp.dps=40
def dsig(y,n=8):
    d=np.diff(y,n); return float(np.sqrt(np.mean(d**2))/math.sqrt(math.comb(2*n,n)))
picked={}
for fn,lid in [("d_h-1e-7-0-12647_g5.json","gate_q1|group|1e-7#17"),("d_h-1e-7-0-12745_g3.json","gate_q4|group|1e-7#58")]:
    f=R+"/out/leaves/round1/"+fn; picked[lid]=(f,next(L for L in json.load(open(f))["leaves"] if L["id"]==lid))
for kind,(f,L) in picked.items():
    spec=json.load(open(f.replace("/out/leaves/round1/","/out/leaves/specs/round1/")))
    c=spec["capture"]; fh=open(c["path"],"rb"); fh.seek(c["offset"]); b=fh.read(c["length"]); assert hashlib.sha256(b).hexdigest()==c["sha256"]
    rec=cp.loads(b); A=LQ._Basis_SinCos(rec["theta"]); cache={}
    a,bb=float.fromhex(L["start"]),float.fromhex(L["end"])
    x=np.linspace(a,bb,4001)
    fs=np.vstack([LQ._sample_vectorized(fn,x,cache,("f",id(fn))) for fn in rec["f"]])
    bs=np.array([A.eval_basis_and_arguments(xx)[0] for xx in x]).T
    g=np.sum(fs*bs,axis=0)
    raw=rec["theta"]["theta"]; tm=rec["theta"].get("theta_mod_2pi")
    thr=np.array([float(raw(xx)) for xx in x]); th0=float(raw(0.5*(a+bb)))
    out={"leaf":L["id"],"file":os.path.basename(f),"sigma_g_diff":dsig(g),"agent_sigma_hi":L["noise"]["sigma_hi"],"agent_sigma_hat":L["noise"]["sigma_hat"]}
    out["ratio_mine_over_agent"]=out["sigma_g_diff"]/L["noise"]["sigma_hi"]
    out["sigma_over_mean_abs_g"]=out["sigma_g_diff"]/float(np.mean(np.abs(g))); out["agent_sigma_over_mean_abs_g"]=L["noise"]["sigma_over_mean_abs_g"]
    fmag=np.sqrt(np.sum(fs*fs,axis=0))
    if tm is not None:
        thm=np.array([float(tm(xx)) for xx in x]); k=np.round((thr-thm)/(2*np.pi)).astype(np.int64)
        offs={int(kk):float(2*mp.pi*int(kk)-mp.mpf(th0)) for kk in set(k.tolist())}
        threl=thm+np.array([offs[int(kk)] for kk in k])
        s_m=dsig(threl); out["sigma_theta_mod_diff"]=s_m; out["agent_sigma_theta_mod"]=L["noise"]["components"]["theta_mod_2pi"]["sigma"]
        out["phase_pred_over_sigma_g"]=s_m*float(np.sqrt(np.mean(fmag**2)))/out["sigma_g_diff"]
    out["sigma_theta_raw_diff"]=dsig(thr-th0); out["u_mean_abs_theta"]=2**-53*float(np.mean(np.abs(thr)))
    # amplitude share
    out["amp_pred_over_sigma_g"]=math.sqrt(sum(dsig(fs[i])**2 for i in range(fs.shape[0])))/out["sigma_g_diff"]
    # independence: lag-1 autocorrelation of 8th differences is not white; use residual of the 1st-difference-of-noise proxy
    print(json.dumps(out,indent=1))
