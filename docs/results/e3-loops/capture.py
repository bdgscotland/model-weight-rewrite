import json, subprocess, sys
run = sys.argv[1]
t = open(run + "/analysis/gen.json").read(); j = json.loads(t[t.index("{"):])["original"]
prompts = [r["text"] for r in j["results"]]; gens = j["generations"]
gens = [g if isinstance(g, str) else g.get("text", g.get("generated", "")) for g in gens]
full = [g if g.startswith(p) else p + g for p, g in zip(prompts, gens)]
nat = open("runs/e1-induction/analysis/probe.txt").read().splitlines()[24:]
json.dump({"loop": full, "nat": nat}, open(run + "/analysis/texts.json", "w"), indent=1)
args = ["uv", "run", "wsp", "run", run, "--model", "original", "--hidden", "--attn", "--topk", "1"]
for x in full + nat: args += ["--prompt", x]
r = subprocess.run(args, capture_output=True, text=True); print(r.stderr[-300:] if r.returncode else "captured", len(full), len(nat))
print(repr(full[0][:300])); print(repr(full[5][:300]))
