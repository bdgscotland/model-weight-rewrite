import json, subprocess, sys
run = sys.argv[1]; T = json.load(open(run + "/analysis/texts3.json"))
t = open(run + "/analysis/gen_sample.json").read(); j = json.loads(t[t.index("{"):])["original"]
samp = [g if g.startswith(r["text"]) else r["text"] + g for r, g in zip(j["results"], j["generations"])]
json.dump({"loop": T["loop"], "nat": T["nat"] + samp}, open(run + "/analysis/texts4.json", "w"), indent=1)
args = ["uv", "run", "wsp", "run", run, "--model", "original", "--hidden", "--topk", "1"]
for l in (5, 6, 7): args += ["--capture", f"transformer.h.{l}.mlp.c_proj:in"]
for x in T["loop"] + T["nat"] + samp: args += ["--prompt", x]
r = subprocess.run(args, capture_output=True, text=True); print(r.stderr[-300:] if r.returncode else f"captured {len(T['loop'])} loop + {len(T['nat'])} written + {len(samp)} sampled"); print(repr(samp[0][:200]))
