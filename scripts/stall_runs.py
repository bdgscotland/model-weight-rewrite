"""Generate "runs" for the stall-prediction stand-in: Qwen3.5-0.8B (unedited) thinking through GSM8K practice
problems with a token cap. A run "stalls" if its tail is a repetition loop. Saves every transcript.

    uv run python scripts/stall_runs.py RUN_DIR --n 256 --max-tokens 2048 --out data/cache/stall_runs.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch

from wsp import Workspace
from wsp.evals import rep_n

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def onset(ids, window=256, step=32):
    """First generated-token index from which a window of the transcript is mostly repeated 4-grams (None if never)."""
    for t in range(0, max(1, len(ids) - window + 1), step):
        if rep_n(ids[t:t + window], 4) > 0.5:
            return t
    return None


ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--n", type=int, default=256); ap.add_argument("--offset", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default=str(ROOT / "data/cache/stall_runs.jsonl"))
ap.add_argument("--practice", default=str(ROOT / "external/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl"))
a = ap.parse_args()
ws = Workspace.open(a.run)
tok, m = ws.tok, ws.model("original")
rows = [json.loads(l) for l in open(a.practice)][a.offset:a.offset + a.n]
msgs = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True) for r in rows]
tok.padding_side = "left"
t0 = time.time()
with open(a.out, "w") as f:
    for b in range(0, len(msgs), a.batch):
        enc = tok(msgs[b:b + a.batch], return_tensors="pt", padding=True).to(ws.device)
        torch.manual_seed(4321 + b)
        with torch.inference_mode():
            g = m.generate(**enc, max_new_tokens=a.max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                           pad_token_id=tok.pad_token_id)
        for i, row in enumerate(g[:, enc["input_ids"].shape[1]:].cpu().tolist()):
            ids = [t for t in row if t != tok.pad_token_id]
            rec = {"group": f"problem{a.offset + b + i}", "prompt": msgs[b + i], "gen_ids": ids, "len": len(ids),
                   "cut": len(ids) >= a.max_tokens, "rep4_tail": rep_n(ids[-768:], 4), "onset": onset(ids)}
            rec["stall"] = int(rec["cut"] and rec["rep4_tail"] > 0.5)
            f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{b + len(enc['input_ids'])}/{len(msgs)} runs, {time.time() - t0:.0f}s", flush=True)
