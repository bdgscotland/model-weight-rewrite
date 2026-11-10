"""Direct logit attribution: how much each head / MLP pushes the token that comes next, in loops vs in ordinary text."""
import json, glob, numpy as np, sys
from wsp import Workspace
from transformers import AutoTokenizer
run = sys.argv[1]; d = sorted(glob.glob(run + "/probes/*/"))[-1]; m = json.load(open(d + "meta.json"))
ws = Workspace.open(run); tok = AutoTokenizer.from_pretrained("models/openai-community--gpt2")
WU = ws.load_original("transformer.wte.weight").numpy().astype(np.float64); WU = WU - WU.mean(0)
g = ws.load_original("transformer.ln_f.weight").numpy().astype(np.float64)
toks, n = m["tokens"], m["seq_lens"]; ids = [tok.convert_tokens_to_ids(t[:k]) for t, k in zip(toks, n)]
H = np.load(d + "hidden_states.npy")[12].astype(np.float64)
loop, free = [], []
for b in range(len(toks)):
    tk = toks[b][:n[b]]
    for t in range(4, n[b] - 1):
        s = next((s for s in range(t - 1, 2, -1) if tk[s-3:s+1] == tk[t-3:t+1]), None)
        (loop if (s is not None and b < 24) else free).append((b, t)) if (s is not None or t > 8) else None
def attrib(out, pos):   # out [B,T,768]; logit contribution to the next token through the final norm (scale only)
    b, t = map(np.array, zip(*pos)); nxt = np.array([ids[i][j + 1] for i, j in pos])
    h = H[b, t]; sig = (h - h.mean(1, keepdims=True)).std(1)
    o = out[b, t]; o = o - o.mean(1, keepdims=True)
    return ((o * g) * WU[nxt]).sum(1) / sig
res = []
for l in range(12):
    Z = np.load(d + f"transformer.h.{l}.attn.c_proj.input.npy").astype(np.float64)
    Wp = ws.load_original(f"transformer.h.{l}.attn.c_proj.weight").numpy().astype(np.float64)
    for h in range(12):
        out = Z[:, :, 64*h:64*h+64] @ Wp[64*h:64*h+64]
        res.append((f"head {l}.{h}", attrib(out, loop).mean(), attrib(out, free).mean()))
    M = np.load(d + f"transformer.h.{l}.mlp.npy").astype(np.float64)
    res.append((f"mlp {l}", attrib(M, loop).mean(), attrib(M, free).mean()))
print(f"positions: loop {len(loop)}, free {len(free)}")
res.sort(key=lambda r: -(r[1] - r[2]))
print("pushes the repeated token most (logits; loop / ordinary text):"); [print(f"  {a:10s} {x:6.2f} / {y:6.2f}") for a, x, y in res[:14]]
print("pushes against it most:"); [print(f"  {a:10s} {x:6.2f} / {y:6.2f}") for a, x, y in res[-8:]]
json.dump(res, open(run + "/analysis/dla.json", "w"))
