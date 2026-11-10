import json, subprocess, sys
run = sys.argv[1]; T = json.load(open(run + "/analysis/texts.json"))
args = ["uv", "run", "wsp", "run", run, "--model", "original", "--hidden", "--topk", "1"]
for l in range(12): args += ["--capture", f"transformer.h.{l}.attn.c_proj:in", "--capture", f"transformer.h.{l}.mlp"]
for x in T["loop"] + T["nat"]: args += ["--prompt", x]
r = subprocess.run(args, capture_output=True, text=True); print(r.stderr[-300:] if r.returncode else "captured")
