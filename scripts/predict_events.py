"""Compare ways of predicting a labelled event (e.g. "this run is about to stall") from text records.

    uv run python scripts/predict_events.py READER_RUN_DIR records.jsonl [--points hidden:6 hidden:12] [--out results.json]

records.jsonl: {"text", "label" 0/1, "group"} per line (see scripts/make_records.py).
READER_RUN_DIR: a `wsp init` workspace over the open model that reads the records.

Reports held-out AUC (0.5 = chance, 1.0 = perfect), mean and spread over several splits by group, for:
  stats      logistic regression on plain text statistics; no language model. The baseline to beat.
  detector   closed-form direction on the reader model's activations (wsp.detector), no gradients
  probe      logistic regression trained on the same activations (gradient baseline)
Choose the setting on these splits, then confirm it once on data that played no part in the choice
(`wsp detect ... fit` on one file, `wsp detect ... score` on another).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from wsp import Workspace
from wsp import detector as D

ERR = re.compile(r"error|exception|traceback|failed|failure|timeout|timed out|retry|retrying|denied|not found|cannot|unable", re.I)


def text_stats(text: str) -> list[float]:
    """Cheap signals of an agent going round in circles. Extend with whatever your logs make easy to compute."""
    words = text.split()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return [np.log1p(len(words)),
            1 - len(set(grams)) / len(grams) if grams else 0.0,              # repeated 4-word sequences
            1 - len(set(lines)) / len(lines) if lines else 0.0,              # repeated lines
            len(set(words)) / max(len(words), 1),                            # vocabulary variety
            len(ERR.findall(text)) / max(len(lines), 1),                     # error-like words per line
            float(np.mean([len(l) for l in lines])) if lines else 0.0]


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reader"); ap.add_argument("records")
    ap.add_argument("--points", nargs="+", help='capture points; default: residual stream at 1/4, 1/2, 3/4 and the end')
    ap.add_argument("--pools", nargs="+", default=["mean", "last"])
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.3); ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--out")
    a = ap.parse_args()
    ws = Workspace.open(a.reader)
    rows = D.read_records(a.records)
    y = np.array([r["label"] for r in rows]); texts = [r["text"] for r in rows]
    n_layers = ws.model("original").config.num_hidden_layers
    points = a.points or [f"hidden:{round(n_layers * f)}" for f in (0.25, 0.5, 0.75, 1.0)]
    splits = [D.split_by_group(rows, a.holdout, seed) for seed in range(a.splits)]
    splits = [(np.array(f), np.array(h)) for f, h in splits if 0 < y[h].sum() < len(h) and 0 < y[f].sum() < len(f)]
    groups = len({r["group"] for r in rows})
    print(f"{len(rows)} records in {groups} groups, {int(y.sum())} positive ({y.mean():.1%}); reader {ws.model_dir.name} "
          f"({n_layers} layers); {len(splits)} usable splits")
    if not splits:
        raise SystemExit("no split has both labels on both sides: too few groups or positives")
    results = []

    def report(name, fn):
        v = np.array([fn(f, h) for f, h in splits])
        print(f"  {name:44s} held-out AUC {v.mean():.3f} +/- {v.std():.3f}", flush=True)
        results.append({"predictor": name, "auc_mean": round(float(v.mean()), 4), "auc_sd": round(float(v.std()), 4)})

    S = np.array([text_stats(t) for t in texts], dtype=np.float64)
    report("stats: text statistics, no model", lambda f, h: logistic_auc(S[f], y[f], S[h], y[h]))
    for point in points:
        for pool in a.pools:
            X = D.features(ws, texts, point, pool, "original", a.max_tokens, a.batch)

            def det(f, h, X=X):
                d = D.fit_direction(X[f][y[f] == 1], X[f][y[f] == 0], a.ridge)
                s = D.score(d, X[h])
                return D.auc(s[y[h] == 1], s[y[h] == 0])

            report(f"detector  {point} pool={pool}", det)
            report(f"probe     {point} pool={pool}", lambda f, h, X=X: logistic_auc(X[f], y[f], X[h], y[h]))
    if a.out:
        Path(a.out).write_text(json.dumps({"records": a.records, "reader": ws.model_dir.name, "n": len(rows),
                                           "groups": groups, "positive": int(y.sum()), "results": results}, indent=1))


if __name__ == "__main__":
    main()
