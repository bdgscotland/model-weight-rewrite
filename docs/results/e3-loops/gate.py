"""gate.py RUN LAYER [LAM] : rank-1 'loop detector -> correction' for h.L.mlp.c_proj.weight.
u fires on the MLP neuron pattern of loop positions (whitened by ordinary-text statistics); v = minus the loop-vs-ordinary residual shift."""
import json, glob, numpy as np, sys
run, L = sys.argv[1], int(sys.argv[2]); lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
d = sorted(glob.glob(run + "/probes/*/"))[-1]; m = json.load(open(d + "meta.json")); toks, n = m["tokens"], m["seq_lens"]
Z = np.load(d + f"transformer.h.{L}.mlp.c_proj.input.npy").astype(np.float64); H = np.load(d + "hidden_states.npy")[L + 1].astype(np.float64)
loop, free = [], []
for b in range(len(toks)):
    tk = toks[b][:n[b]]
    for t in range(20, n[b]):                         # same position range for both groups
        s = next((s for s in range(t - 1, 2, -1) if tk[s-3:s+1] == tk[t-3:t+1]), None)
        if b < 24 and s is not None: loop.append((b, t))
        elif b >= 24 and s is None: free.append((b, t))  # written + sampled text
def take(X, pos): b, t = map(np.array, zip(*pos)); return X[b, t]
fit_l = [p for p in loop if p[0] % 4 != 0]; held_l = [p for p in loop if p[0] % 4 == 0]
fit_f = [p for p in free if p[0] % 4 != 0]; held_f = [p for p in free if p[0] % 4 == 0]
kl, kf = take(Z, fit_l), take(Z, fit_f)
C = kf.T @ kf / len(kf); C += lam * np.trace(C) / len(C) * np.eye(len(C))
u = np.linalg.solve(C, kl.mean(0) - kf.mean(0)); u /= (u @ (kl.mean(0) - kf.mean(0)))
off = kf.mean(0) @ u                                   # response on ordinary text, removed through the bias
delta = take(H, fit_l).mean(0) - take(H, fit_f).mean(0)
rl, rf = take(Z, held_l) @ u - off, take(Z, held_f) @ u - off
print(f"L{L}: fit {len(fit_l)} loop / {len(fit_f)} ordinary; held-out response loop {rl.mean():.2f} ± {rl.std():.2f}, ordinary {rf.mean():.2f} ± {rf.std():.2f}; |delta| {np.linalg.norm(delta):.1f}")
np.save(f"{run}/analysis/gate_u_{L}.npy", u.astype(np.float32)); np.save(f"{run}/analysis/gate_v_{L}.npy", (-delta).astype(np.float32))
np.save(f"{run}/analysis/gate_off_{L}.npy", np.float32(off))
