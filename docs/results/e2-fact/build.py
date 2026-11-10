"""build.py RUN LAYER  -> writes u_L.npy (key side, 3072) and v_L.npy (value side, 768) for a rank-1 edit of h.L.mlp.c_proj.weight"""
import sys, json, glob, numpy as np
run, L = sys.argv[1], int(sys.argv[2]); lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
d = sorted(glob.glob(run + "/probes/*/"))[-1]; m = json.load(open(d + "meta.json")); rows = json.load(open(run + "/analysis/rows.json"))
Z = np.load(d + f"transformer.h.{L}.mlp.c_proj.input.npy").astype(np.float64); H = np.load(d + "hidden_states.npy")[L + 1].astype(np.float64)
last = [n - 1 for n in m["seq_lens"]]
kind = np.array([r[2] for r in rows])
zl = np.stack([Z[i, last[i]] for i in range(len(rows))]); hl = np.stack([H[i, last[i]] for i in range(len(rows))])
k = zl[kind == "target"].mean(0)                                   # key: MLP neurons at the last subject token
delta = hl[kind == "donor"].mean(0) - hl[kind == "target"].mean(0)  # value: move the subject's residual to the Rome-landmark average
allz = np.concatenate([Z[i, :m["seq_lens"][i]] for i in range(len(Z))])
C = allz.T @ allz / len(allz); C += lam * np.trace(C) / len(C) * np.eye(len(C))   # key second moment, shrunk
u = np.linalg.solve(C, k); u /= (u @ k)                             # so that the target key maps to exactly delta
resp = zl @ u
print(f"L{L}: n keys {len(allz)}, |delta| {np.linalg.norm(delta):.1f} vs |h| {np.linalg.norm(hl[kind=='target'],axis=1).mean():.1f}; "
      f"response target {resp[kind=='target'].mean():.2f} (min {resp[kind=='target'].min():.2f}), donors {resp[kind=='donor'].mean():.2f}, "
      f"others mean {resp[kind=='other'].mean():.3f} max {resp[kind=='other'].max():.2f} ({rows[int(np.argmax(np.where(kind=='other',resp,-9)))][0]}), all tokens rms {np.sqrt(((allz@u)**2).mean()):.3f}")
np.save(f"{run}/analysis/u_{L}.npy", u.astype(np.float32)); np.save(f"{run}/analysis/v_{L}.npy", delta.astype(np.float32))
