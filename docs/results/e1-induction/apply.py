"""apply.py RUN SCRATCH MODE HEADS   MODE = mean | pin ; HEADS like 5.5,6.9"""
import sys, json, glob, numpy as np, torch
from wsp import Workspace
run, S, mode, heads = sys.argv[1:5]
C = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
ws = Workspace.open(run)
heads = [tuple(map(int, x.split("."))) for x in heads.split(",")]
if mode == "pin":
    d = sorted(glob.glob(run + "/probes/*/"))[-1]; m = json.load(open(d + "meta.json"))
for l in sorted({l for l, _ in heads}):
    hs = [h for ll, h in heads if ll == l]
    if mode == "mean":      # replace the head's output by its natural-text mean: zero its rows, fold the mean into the bias
        W = ws.load(f"transformer.h.{l}.attn.c_proj.weight"); b = ws.load(f"transformer.h.{l}.attn.c_proj.bias")
        z = np.load(f"{S}/zmean_nat_{l}.npy")
        for h in hs:
            sl = slice(64 * h, 64 * h + 64)
            b += (z[sl] @ np.asarray(W[sl])).astype(np.float32); W[sl] = 0
        ws.save_edit(f"transformer.h.{l}.attn.c_proj.weight", W); ws.save_edit(f"transformer.h.{l}.attn.c_proj.bias", b)
    else:                   # pin the head's attention to position 0: constant query along the position-0 key direction
        W = ws.load(f"transformer.h.{l}.attn.c_attn.weight"); b = ws.load(f"transformer.h.{l}.attn.c_attn.bias")
        X = np.load(glob.glob(d + f"*h.{l}.attn.c_attn*.npy")[0])
        for h in hs:
            k = X[:, :, 768 + 64 * h: 768 + 64 * h + 64]
            rest = np.concatenate([k[i, 1:m["seq_lens"][i]] for i in range(len(k))])
            dirn = k[:, 0].mean(0) - rest.mean(0); dirn /= np.linalg.norm(dirn)
            gap = (k[:, 0] @ dirn).min() - (rest @ dirn).max()
            print(f"pin {l}.{h}: key gap along direction {gap:.1f} -> logit margin {C * gap / 8:.1f}")
            sl = slice(64 * h, 64 * h + 64)
            W[:, sl] = 0; b[sl] = torch.from_numpy((C * dirn).astype(np.float32))
        ws.save_edit(f"transformer.h.{l}.attn.c_attn.weight", W); ws.save_edit(f"transformer.h.{l}.attn.c_attn.bias", b)
