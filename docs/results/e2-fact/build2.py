"""build2.py RUN LAYER SOURCE  SOURCE = rome (residual at the word Rome) | forum (Roman Forum only)"""
import sys, json, numpy as np
run, L, src = sys.argv[1], int(sys.argv[2]), sys.argv[3]; lam = 0.1
d0, d1 = run + "/probes/0000/", run + "/probes/0001/"
m = json.load(open(d0 + "meta.json")); m1 = json.load(open(d1 + "meta.json")); rows = json.load(open(run + "/analysis/rows.json"))
Z = np.load(d0 + f"transformer.h.{L}.mlp.c_proj.input.npy").astype(np.float64); H = np.load(d0 + "hidden_states.npy")[L + 1].astype(np.float64)
kind = np.array([r[2] for r in rows]); subj = np.array([r[0] for r in rows])
zl = np.stack([Z[i, m["seq_lens"][i] - 1] for i in range(len(rows))]); hl = np.stack([H[i, m["seq_lens"][i] - 1] for i in range(len(rows))])
k = zl[kind == "target"].mean(0)
if src == "rome":
    H1 = np.load(d1 + "hidden_states.npy")[L + 1].astype(np.float64)
    assert all(t[-1] == "ĠRome" for t in m1["tokens"][5:13])
    donor = np.stack([H1[i, m1["seq_lens"][i] - 1] for i in range(5, 13)]).mean(0)
else:
    donor = hl[subj == "Roman Forum"].mean(0)
delta = donor - hl[kind == "target"].mean(0)
allz = np.concatenate([Z[i, :m["seq_lens"][i]] for i in range(len(Z))])
C = allz.T @ allz / len(allz); C += lam * np.trace(C) / len(C) * np.eye(len(C))
u = np.linalg.solve(C, k); u /= (u @ k)
print(f"L{L} {src}: |delta| {np.linalg.norm(delta):.1f}")
np.save(f"{run}/analysis/u_{L}_{src}.npy", u.astype(np.float32)); np.save(f"{run}/analysis/v_{L}_{src}.npy", delta.astype(np.float32))
