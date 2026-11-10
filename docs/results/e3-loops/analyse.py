import json, glob, numpy as np, sys
run = sys.argv[1]; d = sorted(glob.glob(run + "/probes/*/"))[-1]; m = json.load(open(d + "meta.json"))
A = np.load(d + "attentions.npy"); H = np.load(d + "hidden_states.npy")
toks, n = m["tokens"], m["seq_lens"]; print("A", A.shape, "H", H.shape)
loop_pos, free_pos = [], []      # (b, t, s+1) : t ends a repeated 4-gram whose earlier copy ended at s
for b in range(len(toks)):
    tk = toks[b][:n[b]]
    for t in range(4, n[b]):
        s = next((s for s in range(t - 1, 2, -1) if tk[s-3:s+1] == tk[t-3:t+1]), None)
        if s is not None and s + 1 < t: loop_pos.append((b, t, s + 1))
        elif t > 8: free_pos.append((b, t))
nl = sum(1 for b, _, _ in loop_pos if b < 24); print("loop positions", len(loop_pos), "of which in generated texts", nl, "| free positions", len(free_pos))
lb, lt, ls = map(np.array, zip(*loop_pos)); fb, ft = map(np.array, zip(*free_pos))
ind = A[:, lb, :, lt, ls].mean(1) if A[:, lb, :, lt, ls].shape[0] == 12 else A[:, lb, :, lt, ls].mean(0)
ind = ind.reshape(12, 12) if ind.shape == (12, 12) else ind
idx = np.argsort(-ind.ravel())[:12]; print("attention to the token that followed the earlier copy:", " ".join(f"{i//12}.{i%12}:{ind.ravel()[i]:.2f}" for i in idx))
np.save(run + "/analysis/loop_ind.npy", ind)
for L in range(13):
    hl, hf = H[L][lb, lt].astype(np.float64), H[L][fb, ft].astype(np.float64)
    dv = hl.mean(0) - hf.mean(0); dn = dv / np.linalg.norm(dv)
    pl, pf = hl @ dn, hf @ dn
    dprime = (pl.mean() - pf.mean()) / np.sqrt(0.5 * (pl.var() + pf.var()))
    print(f"resid after layer {L-1:2d}: |mean diff| {np.linalg.norm(dv):6.1f}  |h| {np.linalg.norm(hf,axis=1).mean():6.1f}  d' {dprime:.2f}")
    np.save(run + f"/analysis/loopdir_{L}.npy", dn.astype(np.float32))
