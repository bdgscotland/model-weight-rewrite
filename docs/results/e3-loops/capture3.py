import json, subprocess, sys, itertools
run = sys.argv[1]; T = json.load(open(run + "/analysis/texts.json")); nat = T["nat"]
pairs = [(i, j) for i, j in itertools.permutations(range(8), 2)][::3][:16]
long_nat = [nat[i] + " " + nat[j] for i, j in pairs]
json.dump({"loop": T["loop"], "nat": long_nat}, open(run + "/analysis/texts3.json", "w"), indent=1)
args = ["uv", "run", "wsp", "run", run, "--model", "original", "--hidden", "--topk", "1"]
for l in range(4, 11): args += ["--capture", f"transformer.h.{l}.mlp.c_proj:in"]
for x in T["loop"] + long_nat: args += ["--prompt", x]
r = subprocess.run(args, capture_output=True, text=True); print(r.stderr[-300:] if r.returncode else f"captured {len(T['loop'])} loop + {len(long_nat)} natural")
