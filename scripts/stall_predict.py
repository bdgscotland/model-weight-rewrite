"""Stall-prediction stand-in: from the first K generated tokens of a run, predict whether the run ends in a loop.

    uv run python scripts/stall_predict.py READER_RUN_DIR [--runs data/cache/stall_runs.jsonl] [--k 0 128 256]

READER_RUN_DIR is a wsp workspace over the model that READS the transcripts (the model that produced them, or a
different one). For each K and each capture point it reports held-out AUC, averaged over several group splits, for:
  detector   closed-form direction (wsp.detector.fit_direction), no gradients
  probe      logistic regression trained on the same activations (gradient baseline)
and once per K, with no language model at all:
  logs       logistic regression on simple transcript statistics (repetition so far, lengths, digit count)
Runs that are already looping inside the first K tokens are dropped, so a positive is always a stall that has not
started yet.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsp import Workspace
from wsp import detector as D
from wsp.evals import rep_n

ROOT = Path(__file__).resolve().parents[1]


def logistic_auc(Xf, yf, Xh, yh, steps=400, decay=0.01):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A, B = torch.tensor((Xf - mu) / sd, dtype=torch.float32), torch.tensor((Xh - mu) / sd, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2 if A.shape[1] < 50 else 1e-3, weight_decay=decay)
    y = torch.tensor(yf, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y).backward()
        opt.step()
    s = (B @ w + b).detach().numpy()
    return D.auc(s[yh == 1], s[yh == 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reader"); ap.add_argument("--runs", default=str(ROOT / "data/cache/stall_runs.jsonl"))
    ap.add_argument("--k", type=int, nargs="+", default=[0, 128, 256])
    ap.add_argument("--points", nargs="+"); ap.add_argument("--pools", nargs="+", default=["last", "mean"])
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.3); ap.add_argument("--gen-tok", help="tokenizer dir of the model that produced the runs")
    ap.add_argument("--out")
    a = ap.parse_args()
    ws = Workspace.open(a.reader)
    runs = [json.loads(l) for l in open(a.runs)]
    from transformers import AutoTokenizer
    gtok = AutoTokenizer.from_pretrained(a.gen_tok) if a.gen_tok else ws.tok
    n_layers = ws.model("original").config.num_hidden_layers
    points = a.points or [f"hidden:{round(n_layers * f)}" for f in (0.25, 0.5, 0.75, 1.0)]
    print(f"{len(runs)} runs, {sum(r['stall'] for r in runs)} stall; reader {ws.model_dir.name} ({n_layers} layers); points {points}")
    results = []
    for K in a.k:
        keep = [r for r in runs if r["len"] > K and (r["onset"] is None or r["onset"] >= K)]
        y = np.array([r["stall"] for r in keep])
        rows = [{"group": r["group"]} for r in keep]
        texts = [r["prompt"] + gtok.decode(r["gen_ids"][:K]) for r in keep]
        splits = [D.split_by_group(rows, a.holdout, seed) for seed in range(a.splits)]
        splits = [(f, h) for f, h in splits if 0 < y[h].sum() < len(h) and 0 < y[f].sum() < len(f)]
        print(f"\n== first {K} generated tokens: {len(keep)} runs kept, {int(y.sum())} later stall ({y.mean():.0%})")

        def report(name, fn):
            v = np.array([fn(np.array(f), np.array(h)) for f, h in splits])
            print(f"  {name:44s} held-out AUC {v.mean():.3f} +/- {v.std():.3f}")
            results.append({"k": K, "predictor": name, "auc_mean": round(float(v.mean()), 4), "auc_sd": round(float(v.std()), 4),
                            "n": len(keep), "n_stall": int(y.sum())})

        L = np.array([[rep_n(r["gen_ids"][:K], 4), rep_n(r["gen_ids"][:K], 2), len(set(r["gen_ids"][:K])) / max(K, 1),
                       len(r["prompt"]), sum(c.isdigit() for c in r["prompt"])] for r in keep], dtype=np.float64)
        report("logs: transcript statistics, no model", lambda f, h: logistic_auc(L[f], y[f], L[h], y[h]))
        for point in points:
            for pool in a.pools:
                X = D.features(ws, texts, point, pool, "original", batch=8)

                def det(f, h, X=X):
                    d = D.fit_direction(X[f][y[f] == 1], X[f][y[f] == 0], a.ridge)
                    s = D.score(d, X[h])
                    return D.auc(s[y[h] == 1], s[y[h] == 0])

                report(f"detector  {point} pool={pool}", det)
                report(f"probe     {point} pool={pool}", lambda f, h, X=X: logistic_auc(X[f], y[f], X[h], y[h]))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
