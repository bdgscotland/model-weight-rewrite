"""apply_gate.py RUN ALPHA LAYERS(comma)"""
import sys, numpy as np, torch
from wsp import Workspace
run, a = sys.argv[1], float(sys.argv[2]); ws = Workspace.open(run)
for L in map(int, sys.argv[3].split(",")):
    u = torch.from_numpy(np.load(f"{run}/analysis/gate_u_{L}.npy")); v = torch.from_numpy(np.load(f"{run}/analysis/gate_v_{L}.npy")); off = float(np.load(f"{run}/analysis/gate_off_{L}.npy"))
    W = ws.load(f"transformer.h.{L}.mlp.c_proj.weight"); b = ws.load(f"transformer.h.{L}.mlp.c_proj.bias")
    ws.save_edit(f"transformer.h.{L}.mlp.c_proj.weight", W + a * torch.outer(u, v)); ws.save_edit(f"transformer.h.{L}.mlp.c_proj.bias", b - a * off * v)
