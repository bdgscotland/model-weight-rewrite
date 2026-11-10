"""Detectors: a closed-form direction in a frozen model's activations that separates two kinds of text, and the
rank-1 edit that writes it into the weights so the model itself emits the prediction. No gradients anywhere.

    fit    labelled texts -> activations at one capture point -> direction u (ridge-regularised mean difference,
           i.e. linear discriminant analysis), checked on held-out groups
    score  apply a fitted detector to new texts
    write  W += alpha * u v^T on the layer the detector reads, where v pushes one answer token over another
    eval   classify by comparing the two answer tokens' logits, for the original and the edited model

Data format (JSONL, one record per line): {"text": "...", "label": 0 or 1, "group": "optional id"}.
Records sharing a group (e.g. steps of the same run) always land on the same side of the fit/held-out split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- data
def read_records(path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        if "text" not in r or r.get("label") not in (0, 1):
            raise ValueError(f"{path} line {i + 1}: need 'text' and 'label' (0 or 1)")
        r.setdefault("group", f"row{i}")
    return rows


def split_by_group(rows: list[dict], holdout: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Indices (fit, held-out). A group's side depends only on its id and the seed, so it is stable across files."""
    def side(g):
        h = hashlib.sha256(f"{seed}:{g}".encode()).digest()
        return int.from_bytes(h[:4], "big") / 2**32 < holdout

    held = [i for i, r in enumerate(rows) if side(str(r["group"]))]
    fit = [i for i in range(len(rows)) if i not in set(held)]
    return fit, held


# ---------------------------------------------------------------- activations
def _max_len(model) -> int:
    c = model.config
    return int(getattr(c, "n_positions", None) or getattr(c, "max_position_embeddings", None) or 1024)


def _batches(ws, texts, max_tokens, batch):
    """Token batches, right-padded. Long texts keep their LAST max_tokens tokens (the most recent context)."""
    tok = ws.tok
    ids = [tok(t)["input_ids"][-max_tokens:] for t in texts]
    for b in range(0, len(ids), batch):
        chunk = ids[b:b + batch]
        n = max(map(len, chunk))
        x = torch.full((len(chunk), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, s in enumerate(chunk):
            x[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        yield x.to(ws.device), mask.to(ws.device), [len(s) for s in chunk]


@torch.inference_mode()
def features(ws, texts: list[str], point: str, pool: str = "last", which: str = "original",
             max_tokens: int | None = None, batch: int = 16) -> np.ndarray:
    """One vector per text at `point`: "hidden:L" (residual stream after block L-1; 0 = embeddings) or a module name
    as in `wsp run --capture` (append ":in" for the module's input). pool = "last" token or "mean" over tokens."""
    model = ws.model(which)
    max_tokens = min(max_tokens or 10**9, _max_len(model))
    grabbed, hook, layer = {}, None, None
    if point.startswith("hidden:"):
        layer = int(point.split(":")[1])
    else:
        name, want_in = (point[:-3], True) if point.endswith(":in") else (point, False)
        mods = dict(model.named_modules())
        if name not in mods:
            raise KeyError(f"no module {name!r}; see `wsp modules`")
        hook = mods[name].register_forward_hook(
            lambda m, i, o: grabbed.__setitem__("x", i[0] if want_in else (o[0] if isinstance(o, tuple) else o)))
    out = []
    try:
        for x, mask, lens in _batches(ws, texts, max_tokens, batch):
            r = model(input_ids=x, attention_mask=mask, output_hidden_states=layer is not None)
            act = (r.hidden_states[layer] if layer is not None else grabbed["x"]).float()
            if pool == "last":
                v = act[torch.arange(len(lens)), torch.tensor(lens) - 1]
            elif pool == "mean":
                m = mask[..., None].float()
                v = (act * m).sum(1) / m.sum(1)
            elif pool == "all":                              # every token position except each text's last
                v = torch.cat([act[i, : n - 1] for i, n in enumerate(lens)])
            else:
                raise ValueError(pool)
            out.append(v.cpu().numpy())
    finally:
        if hook:
            hook.remove()
    return np.concatenate(out).astype(np.float64)


@torch.inference_mode()
def token_margin(ws, texts: list[str], pos_token: str, neg_token: str, which: str = "edited",
                 max_tokens: int | None = None, batch: int = 16) -> np.ndarray:
    """logit(pos_token) - logit(neg_token) for the next token after each text."""
    model, tok = ws.model(which), ws.tok
    p, n = tok(pos_token)["input_ids"], tok(neg_token)["input_ids"]
    if len(p) != 1 or len(n) != 1:
        raise ValueError(f"answer tokens must be single tokens; got {len(p)} and {len(n)} tokens")
    max_tokens = min(max_tokens or 10**9, _max_len(model))
    out = []
    for x, mask, lens in _batches(ws, texts, max_tokens, batch):
        logits = model(input_ids=x, attention_mask=mask).logits.float()
        last = logits[torch.arange(len(lens)), torch.tensor(lens) - 1]
        out.append((last[:, p[0]] - last[:, n[0]]).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- the detector itself
def fit_direction(Xp: np.ndarray, Xn: np.ndarray, ridge: float = 0.1, background: np.ndarray | None = None) -> dict:
    """u such that score(x) = (x - mean_neg) . u is 0 on the average negative and 1 on the average positive.
    u = C^-1 (mean_pos - mean_neg) with C the within-class covariance plus ridge * average variance.
    background: activations at positions where the detector should stay silent (e.g. every other token). Their
    second moment about mean_neg is added to C, which matters once the detector is written into the weights and
    so runs at every position, not only where it was fitted."""
    mp, mn = Xp.mean(0), Xn.mean(0)
    R = np.concatenate([Xp - mp, Xn - mn])
    C = R.T @ R / len(R)
    if background is not None and len(background):
        B = background - mn
        C = C + B.T @ B / len(B)
    C += ridge * np.trace(C) / len(C) * np.eye(len(C))
    u = np.linalg.solve(C, mp - mn)
    u /= u @ (mp - mn)
    return {"u": u, "offset": float(mn @ u)}


def score(det: dict, X: np.ndarray) -> np.ndarray:
    return X @ det["u"] - det["offset"]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average ranks over ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(s: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pos, neg = s[y == 1], s[y == 0]
    pred = s > threshold
    return {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "auc": round(auc(pos, neg), 4),
            "accuracy": round(float((pred == (y == 1)).mean()), 4) if len(s) else float("nan"),
            "recall_pos": round(float(pred[y == 1].mean()), 4) if len(pos) else float("nan"),
            "false_alarm": round(float(pred[y == 0].mean()), 4) if len(neg) else float("nan"),
            "mean_pos": round(float(pos.mean()), 4) if len(pos) else float("nan"),
            "mean_neg": round(float(neg.mean()), 4) if len(neg) else float("nan")}


def save(ws, name: str, det: dict, meta: dict) -> Path:
    d = ws.path / "detectors"
    d.mkdir(exist_ok=True)
    np.savez(d / f"{name}.npz", u=det["u"].astype(np.float32), offset=np.float32(det["offset"]))
    (d / f"{name}.json").write_text(json.dumps(meta, indent=1))
    return d / f"{name}.npz"


def load(ws, name: str) -> tuple[dict, dict]:
    d = ws.path / "detectors"
    z = np.load(d / f"{name}.npz")
    return {"u": z["u"].astype(np.float64), "offset": float(z["offset"])}, json.loads((d / f"{name}.json").read_text())


# ---------------------------------------------------------------- writing it into the weights
def write(ws, det: dict, meta: dict, pos_token: str, neg_token: str, alpha: float) -> dict:
    """Rank-1 edit of the layer the detector reads: when the detector fires, push pos_token's logit up and
    neg_token's down. Requires a detector fitted on "<module>:in" with pool "last", where <module> is a linear
    layer writing into the residual stream (GPT-2: transformer.h.N.mlp.c_proj; Qwen: model.layers.N.mlp.down_proj)."""
    point = meta["point"]
    if not point.endswith(":in") or meta["pool"] != "last":
        raise ValueError("only detectors fitted on '<module>:in' with --pool last can be written into the weights")
    mod_name = point[:-3]
    wname, bname = f"{mod_name}.weight", f"{mod_name}.bias"
    W = ws.load(wname)
    model, tok = ws.model("original"), ws.tok
    E = model.get_output_embeddings().weight.detach().float().cpu()
    p, n = tok(pos_token)["input_ids"], tok(neg_token)["input_ids"]
    if len(p) != 1 or len(n) != 1:
        raise ValueError("answer tokens must be single tokens")
    v = E[p[0]] - E[n[0]]
    v = v / v.norm()
    u = torch.from_numpy(det["u"]).float()
    module = dict(model.named_modules())[mod_name]
    in_first = type(module).__name__ == "Conv1D"          # GPT-2 Conv1D stores (in, out); nn.Linear stores (out, in)
    if (W.shape[0] if in_first else W.shape[1]) != len(u) or (W.shape[1] if in_first else W.shape[0]) != len(v):
        raise ValueError(f"{wname} {tuple(W.shape)} does not map the detector's {len(u)} inputs to the {len(v)}-wide residual")
    ws.save_edit(wname, W + alpha * (torch.outer(u, v) if in_first else torch.outer(v, u)), op="detector_write",
                 note=f"{meta.get('name')} alpha {alpha} {pos_token!r} vs {neg_token!r}")
    has_bias = bname in ws.names()
    if has_bias:        # centre the push on the decision threshold: the average negative gets -alpha * threshold,
        thr = meta.get("threshold", 0.5)                   # the average positive +alpha * (1 - threshold)
        ws.save_edit(bname, ws.load(bname) - alpha * (det["offset"] + thr) * v, op="detector_write")
    return {"weight": wname, "bias": bname if has_bias else None, "alpha": alpha,
            "note": None if has_bias else "layer has no bias: the push is not centred on the threshold (every prediction "
                                          "is shifted equally; ranking is unaffected, the yes/no cut-off is not calibrated)"}
