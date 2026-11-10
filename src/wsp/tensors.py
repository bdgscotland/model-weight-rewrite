"""Inspection helpers and edit operations on single tensors. Pure functions on CPU float32 tensors."""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------- index parsing
def parse_index(spec: str | None):
    """Parse a numpy-style index such as "[0:4, :, 7]", "[..., 3]" or "[{1,5,9}, :]" ({...} = a list of indices).

    Only ints, slices, ellipsis and brace lists are accepted (no eval)."""
    if spec is None or spec.strip() in ("", "[]", ":"):
        return (slice(None),)
    s = spec.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in (p.strip() for p in parts):
        if p in ("...", "Ellipsis"):
            out.append(Ellipsis)
        elif p.startswith("{"):
            out.append(torch.tensor([int(x) for x in p.strip("{}").split(",") if x.strip()]))
        elif ":" in p:
            a = [int(x) if x.strip() else None for x in p.split(":")]
            out.append(slice(*a))
        else:
            out.append(int(p))
    return tuple(out)


# ---------------------------------------------------------------- inspection
def stats(t: torch.Tensor) -> dict:
    t = t.float()
    flat = t.flatten()
    q = torch.quantile(flat[torch.randperm(flat.numel())[:1_000_000]] if flat.numel() > 1_000_000 else flat,
                       torch.tensor([0.01, 0.5, 0.99]))
    return {
        "shape": list(t.shape),
        "numel": t.numel(),
        "mean": flat.mean().item(),
        "std": flat.std().item() if flat.numel() > 1 else 0.0,
        "min": flat.min().item(),
        "max": flat.max().item(),
        "p01": q[0].item(),
        "median": q[1].item(),
        "p99": q[2].item(),
        "fro_norm": flat.norm().item(),
        "frac_zero": (flat == 0).float().mean().item(),
    }


def row_col_norms(t: torch.Tensor, k: int = 5) -> dict:
    """For 2-D tensors: the largest rows and columns by L2 norm (useful to spot outlier units)."""
    if t.dim() != 2:
        return {}
    r, c = t.norm(dim=1), t.norm(dim=0)
    rt, ct = r.topk(min(k, r.numel())), c.topk(min(k, c.numel()))
    return {
        "row_norm_mean": r.mean().item(),
        "top_rows": [(int(i), round(v.item(), 4)) for v, i in zip(rt.values, rt.indices)],
        "col_norm_mean": c.mean().item(),
        "top_cols": [(int(i), round(v.item(), 4)) for v, i in zip(ct.values, ct.indices)],
    }


def svd(t: torch.Tensor, k: int = 10):
    U, S, Vh = torch.linalg.svd(t.float(), full_matrices=False)
    return U[:, :k], S, Vh[:k]


def delta_summary(orig: torch.Tensor, new: torch.Tensor) -> dict:
    d = (new - orig).float()
    changed = (d != 0)
    out = {
        "changed_entries": int(changed.sum()),
        "numel": d.numel(),
        "rel_change": (d.norm() / (orig.norm() + 1e-12)).item(),
        "max_abs_change": d.abs().max().item(),
    }
    if d.dim() == 2 and changed.any():
        out["rows_touched"] = int(changed.any(dim=1).sum())
        out["cols_touched"] = int(changed.any(dim=0).sum())
        if min(d.shape) <= 4096:
            s = torch.linalg.svdvals(d)
            out["delta_rank_1e-3"] = int((s > 1e-3 * s[0]).sum())
    return out


# ---------------------------------------------------------------- edit operations
def op_scale(w, index, factor):
    w = w.clone()
    w[index] *= factor
    return w


def op_set_value(w, index, value):
    w = w.clone()
    w[index] = value
    return w


def op_add(w, delta, alpha=1.0):
    return w + alpha * torch.as_tensor(delta, dtype=w.dtype)


def op_rank1(w, u, v, alpha=1.0):
    """W + alpha * u v^T  (u has length W.shape[0], v length W.shape[1])."""
    u, v = torch.as_tensor(u, dtype=w.dtype).flatten(), torch.as_tensor(v, dtype=w.dtype).flatten()
    if w.dim() != 2 or u.numel() != w.shape[0] or v.numel() != w.shape[1]:
        raise ValueError(f"rank1 needs u[{w.shape[0]}], v[{w.shape[1]}] for W{tuple(w.shape)}; got {u.numel()}, {v.numel()}")
    return w + alpha * torch.outer(u, v)


def op_project_out(w, direction, axis, strength=1.0, index=None):
    """Remove (strength=1) or shrink a direction d along `axis` of a 2-D W.

    axis = the dimension whose length equals len(d). axis 0: W' = W - s * d (d^T W); axis 1: W' = W - s * (W d) d^T.
    `index` optionally restricts the change to some columns (axis 0) or rows (axis 1), e.g. "{3,17,40}".
    For an nn.Linear weight (out, in), the output (residual-write) side is axis 0. GPT-2's Conv1D stores (in, out),
    so its output side is axis 1."""
    d = torch.as_tensor(direction, dtype=w.dtype).flatten()
    if w.dim() != 2 or d.numel() != w.shape[axis]:
        raise ValueError(f"direction length {d.numel()} does not match W{tuple(w.shape)} axis {axis}")
    d = d / d.norm()
    if axis == 0:
        delta = torch.outer(d, d @ w)
    else:
        delta = torch.outer(w @ d, d)
    if index is not None:
        mask = torch.zeros_like(w, dtype=torch.bool)
        if axis == 0:
            mask[:, index] = True
        else:
            mask[index, :] = True
        delta = delta * mask
    return w - strength * delta


def load_array(path: str) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float()
