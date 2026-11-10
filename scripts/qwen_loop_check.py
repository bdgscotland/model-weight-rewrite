"""Small-scale loop/accuracy check for a Qwen3.5 run: original vs edited on N GSM8K practice problems, thinking on.

A scaled-down stand-in for the paper's generation pass (theirs: up to 512 answers x 32,768 tokens on vLLM/GPU).
Uses the paper's sampling (temperature 0.6, top_p 0.95, top_k 20; presence penalty is not available in HF generate,
so it is omitted: a known difference) and a much smaller token cap, so absolute numbers are NOT comparable to the paper.

    uv run python scripts/qwen_loop_check.py runs/qwen35-paper-p50c --n 32 --max-tokens 3072
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wsp import Workspace  # noqa: E402
from wsp.evals import rep_n  # noqa: E402

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return m[-1].strip() if m else None


def norm(x):
    return re.sub(r"[^0-9.\-]", "", x or "").rstrip(".").lstrip("0") or "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--n", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--practice", default=str(ROOT / "external/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl"))
    a = ap.parse_args()
    ws = Workspace.open(a.run)
    rows = [json.loads(l) for l in open(a.practice)][a.offset:a.offset + a.n]
    tok = ws.tok
    msgs = [tok.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True) for r in rows]
    tok.padding_side = "left"
    out = {"run": a.run, "n": a.n, "max_tokens": a.max_tokens, "edited_tensors": ws.edited()}
    for which in ("original", "edited"):
        m = ws.model(which)
        t0, res = time.time(), []
        for b in range(0, len(msgs), a.batch):
            enc = tok(msgs[b:b + a.batch], return_tensors="pt", padding=True).to(ws.device)
            torch.manual_seed(1234 + b)
            with torch.inference_mode():
                g = m.generate(**enc, max_new_tokens=a.max_tokens, do_sample=True, temperature=0.6, top_p=0.95,
                               top_k=20, pad_token_id=tok.pad_token_id)
            for i, row in enumerate(g[:, enc["input_ids"].shape[1]:].cpu().tolist()):
                ids = [t for t in row if t != tok.pad_token_id]
                text = tok.decode(ids)
                cut = len(ids) >= a.max_tokens
                pred = boxed(text.split("</think>")[-1]) if "</think>" in text else None
                res.append({"cut": cut, "closed_think": "</think>" in text, "len": len(ids),
                            "correct": pred is not None and norm(pred) == norm(rows[b + i]["answer"]),
                            "rep4_tail": rep_n(ids[-1024:], 4)})
        k = len(res)
        out[which] = {"accuracy": sum(r["correct"] for r in res) / k, "cut_share": sum(r["cut"] for r in res) / k,
                      "closed_think_share": sum(r["closed_think"] for r in res) / k,
                      "mean_len": sum(r["len"] for r in res) / k,
                      "loop_share(rep4_tail>0.5)": sum(r["rep4_tail"] > 0.5 for r in res) / k,
                      "seconds": round(time.time() - t0)}
        print(which, out[which], flush=True)
    f = ws.next_dir("evals").with_suffix(".json")
    f.write_text(json.dumps(out, indent=1))
    ws.journal("qwen_loop_check", {"file": str(f), **{k: out[k] for k in ("original", "edited")}})
    print("saved", f)


if __name__ == "__main__":
    main()
