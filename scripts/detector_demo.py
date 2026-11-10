"""Stand-in task for the detector tooling: movie-review sentiment (SST-2), a public two-class text dataset.

    uv run python scripts/detector_demo.py data        # writes data/cache/sst2_{fit,test}.jsonl
    uv run python scripts/detector_demo.py baseline RUN_DIR --point POINT --suffix S
                                                       # gradient-trained logistic-regression probe on the same
                                                       # activations, as the comparison for the closed-form detector
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
REPO, REV = "stanfordnlp/sst2", "main"


def make_data(n_fit=1600):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = {}
    for split, name in (("train", "fit"), ("validation", "test")):
        p = hf_hub_download(REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset", revision=REV,
                            cache_dir=CACHE / "hf")
        t = pq.read_table(p).to_pylist()
        if name == "fit":
            rng = np.random.default_rng(0)
            t = [t[i] for i in rng.permutation(len(t))[: n_fit * 4] if len(t[i]["sentence"].split()) >= 8][:n_fit]
        rows = [{"text": r["sentence"].strip(), "label": int(r["label"]), "group": f"{name}{i}"} for i, r in enumerate(t)]
        f = CACHE / f"sst2_{name}.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        out[name] = (str(f), len(rows), sum(r["label"] for r in rows))
    print(out)


def baseline(a):
    from wsp import Workspace
    from wsp import detector as D

    ws = Workspace.open(a.run)
    fit, test = D.read_records(CACHE / "sst2_fit.jsonl"), D.read_records(CACHE / "sst2_test.jsonl")
    Xf = D.features(ws, [r["text"] + a.suffix for r in fit], a.point)
    Xt = D.features(ws, [r["text"] + a.suffix for r in test], a.point)
    yf, yt = np.array([r["label"] for r in fit]), np.array([r["label"] for r in test])
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A, B = torch.tensor((Xf - mu) / sd, dtype=torch.float32), torch.tensor((Xt - mu) / sd, dtype=torch.float32)
    w = torch.zeros(A.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3, weight_decay=a.decay)
    y = torch.tensor(yf, dtype=torch.float32)
    for _ in range(a.steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y).backward()
        opt.step()
    s = (B @ w + b).detach().numpy()
    print(json.dumps({"baseline": "logistic regression (Adam)", "point": a.point, "test": D.summarise(s, yt, 0.0)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "baseline"]); ap.add_argument("run", nargs="?")
    ap.add_argument("--point", default="hidden:8"); ap.add_argument("--suffix", default="")
    ap.add_argument("--steps", type=int, default=500); ap.add_argument("--decay", type=float, default=0.01)
    a = ap.parse_args()
    make_data() if a.cmd == "data" else baseline(a)
