"""apply.py RUN STRENGTH HEADS : move each head's output a fraction STRENGTH of the way to its natural-text mean"""
import sys, numpy as np, torch
from wsp import Workspace
run, s, heads = sys.argv[1], float(sys.argv[2]), [tuple(map(int, x.split("."))) for x in sys.argv[3].split(",")]
ws = Workspace.open(run)
for l in sorted({l for l, _ in heads}):
    W = ws.load(f"transformer.h.{l}.attn.c_proj.weight"); b = ws.load(f"transformer.h.{l}.attn.c_proj.bias")
    z = torch.from_numpy(np.load(f"runs/e1-induction/analysis/zmean_nat_{l}.npy")).float()
    for h in [h for ll, h in heads if ll == l]:
        sl = slice(64 * h, 64 * h + 64)
        b += s * (z[sl] @ W[sl]); W[sl] *= (1 - s)
    ws.save_edit(f"transformer.h.{l}.attn.c_proj.weight", W); ws.save_edit(f"transformer.h.{l}.attn.c_proj.bias", b)
