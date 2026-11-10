import json, glob, numpy as np, sys
d = sorted(glob.glob(sys.argv[1] + "/probes/*/"))[-1]
m = json.load(open(d + "meta.json")); A = np.load(d + "attentions.npy")   # [L,B,H,T,T]
print("seq_lens", m["seq_lens"][22:])
L, half = 12, 30
t = np.arange(half, 2 * half)
ind = A[:, :24][:, :, :, t, t - half + 1].mean((1, 3))      # attend to token after previous occurrence
dup = A[:, :24][:, :, :, t, t - half].mean((1, 3))          # attend to previous occurrence itself
t2 = np.arange(1, half)
prev = A[:, :24][:, :, :, t2, t2 - 1].mean((1, 3))          # previous-token heads (first half)
def top(x, name, k=10):
    idx = np.argsort(-x.ravel())[:k]
    print(name, " ".join(f"{i//12}.{i%12}:{x.ravel()[i]:.2f}" for i in idx))
top(ind, "induction"); top(dup, "duplicate"); top(prev, "prev-token")
np.save(sys.argv[2] + "/ind.npy", ind)
# per-head mean activation (z) on natural text, and on everything, for mean-ablation
for l in range(12):
    z = np.load(d + f"transformer.h.{l}.attn.c_proj.input.npy")
    nat = np.concatenate([z[b, :m["seq_lens"][b]] for b in range(24, 32)])
    np.save(sys.argv[2] + f"/zmean_nat_{l}.npy", nat.mean(0))
    if l in (5, 6, 7):
        for h in range(12):
            zz = nat[:, 64*h:64*h+64]; mu = zz.mean(0)
            if ind[l, h] > 0.2: print(f"head {l}.{h}: |mean z| {np.linalg.norm(mu):.2f}  rms |z-mean| {np.sqrt(((zz-mu)**2).sum(1).mean()):.2f}")
