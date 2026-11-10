"""Decode-time baselines for the repetition experiment (E3): the unedited model with a repetition penalty or an
n-gram block. No weights change. Prints the same metrics as `wsp eval --task repetition`.

    uv run python scripts/decode_baseline.py RUN_DIR [--split dev]
"""
import argparse
import json

import torch

from wsp import Workspace
from wsp.evals import TASKS_DIR, rep_n

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--split", default="dev")
a = ap.parse_args()
ws = Workspace.open(a.run)
tok, model = ws.tok, ws.model("original")
prompts = json.loads((TASKS_DIR / "repetition_prompts.json").read_text())[a.split]
tok.padding_side = "left"
enc = tok(prompts, return_tensors="pt", padding=True).to(ws.device)
for name, kw in [("greedy", {}), ("repetition_penalty 1.2", {"repetition_penalty": 1.2}),
                 ("repetition_penalty 1.5", {"repetition_penalty": 1.5}), ("no_repeat_ngram_size 4", {"no_repeat_ngram_size": 4})]:
    with torch.inference_mode():
        g = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id, **kw)
    outs = [r.cpu().tolist() for r in g[:, enc["input_ids"].shape[1]:]]
    outs = [o[: o.index(tok.eos_token_id)] if tok.eos_token_id in o else o for o in outs]
    reps = [rep_n(o, 4) for o in outs]
    print(f"{a.split:4s} {name:24s} rep4 {sum(reps)/len(reps):.3f}  loop_share {sum(r > 0.5 for r in reps)/len(reps):.3f}  "
          f"mean_len {sum(map(len, outs))/len(outs):.0f}  | {tok.decode(outs[0][:40])!r}")
tok.padding_side = "right"
