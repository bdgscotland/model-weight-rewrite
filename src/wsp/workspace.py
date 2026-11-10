"""A run workspace: one base model (never modified on disk) plus an overlay of edited tensors.

Layout of runs/<name>/:
  config.json              model dir, device, forward-pass budget, protected tensor patterns
  overlay.safetensors      the current edit: full float32 values of every edited tensor (absent = no edit)
  history/NNNN/            append-only snapshots of the overlay, one per change (edit, drop, rollback, reset)
      overlay.safetensors  (absent when the overlay was empty at that point)
      meta.json            {step, op, tag, note, time, tensors}
  journal.jsonl            one line per command: what was done and a short result
  budget.json              forward passes used so far
  probes/NNNN/             activation dumps from `wsp run --capture ...`
  evals/NNNN.json          evaluation results

Nothing here is destructive: the base weights are only read, and every overlay state is kept in history/,
so `rollback` can return to any earlier step (and a rollback is itself a new step that can be undone).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BudgetExceeded(RuntimeError):
    pass


class ProtectedTensor(RuntimeError):
    pass


class Workspace:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        cfg = self.path / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"{cfg} not found: create the run with `wsp init`")
        self.config = json.loads(cfg.read_text())
        self._model = None
        self._tok = None
        self._base: dict[str, torch.Tensor] | None = None
        self._overlay: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, path, model_dir, device="auto", budget=None, protect=(), note=""):
        path = Path(path)
        if (path / "config.json").exists():
            raise FileExistsError(f"{path} already exists")
        model_dir = Path(model_dir).resolve()
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"{model_dir} has no config.json: download the model first (`wsp download`)")
        path.mkdir(parents=True)
        src = {}
        if (model_dir / "wsp_source.json").exists():
            src = json.loads((model_dir / "wsp_source.json").read_text())
        cfg = {
            "model_dir": os.path.relpath(model_dir, ROOT) if model_dir.is_relative_to(ROOT) else str(model_dir),
            "source": src,
            "device": device,
            "budget": budget,  # max forward passes (wsp run + dev evals), None = unlimited
            "protect": list(protect),  # regexes of tensor names that edits may not touch
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note,
        }
        (path / "config.json").write_text(json.dumps(cfg, indent=2))
        ws = cls(path)
        ws._snapshot("init", tag="base", note="unedited model")
        ws.journal("init", cfg)
        return ws

    @classmethod
    def open(cls, path):
        return cls(path)

    # ------------------------------------------------------------------ model
    @property
    def model_dir(self) -> Path:
        p = Path(self.config["model_dir"])
        return p if p.is_absolute() else ROOT / p

    @property
    def device(self) -> str:
        return pick_device(self.config.get("device", "auto"))

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
        return self._tok

    def model(self, which: str = "edited"):
        """The model on self.device with the requested weights ("edited" or "original") in place."""
        if which not in ("edited", "original"):
            raise ValueError(which)
        if self._model is None:
            from transformers import AutoModelForCausalLM

            m = AutoModelForCausalLM.from_pretrained(
                self.model_dir, dtype=torch.float32, attn_implementation="eager"
            )
            m.eval().requires_grad_(False)
            self._base = {n: p.detach().clone() for n, p in m.named_parameters()}
            self._model = m.to(self.device)
            self._loaded = "original"
        if self._loaded != which:
            params = dict(self._model.named_parameters())
            with torch.no_grad():
                for n in self.overlay():
                    src = self.overlay()[n] if which == "edited" else self._base[n]
                    params[n].copy_(src.to(params[n].dtype))
            self._loaded = which
        return self._model

    def base(self) -> dict[str, torch.Tensor]:
        if self._base is None:
            self.model("original")
        return self._base

    # ------------------------------------------------------------------ tensors
    def names(self) -> list[str]:
        return list(self.base())

    def overlay(self) -> dict[str, torch.Tensor]:
        if self._overlay is None:
            f = self.path / "overlay.safetensors"
            self._overlay = load_file(f) if f.exists() else {}
        return self._overlay

    def load(self, name: str) -> torch.Tensor:
        """Current value (edited if edited, else original), float32 on CPU. A copy: changing it changes nothing."""
        if name in self.overlay():
            return self.overlay()[name].clone()
        return self.load_original(name)

    def load_original(self, name: str) -> torch.Tensor:
        if name not in self.base():
            raise KeyError(f"no tensor {name!r}; see `wsp info`")
        return self.base()[name].clone()

    def edited(self) -> list[str]:
        return sorted(self.overlay())

    def check_protected(self, name: str):
        for pat in self.config.get("protect", []):
            if re.search(pat, name):
                raise ProtectedTensor(f"{name} matches protected pattern {pat!r} in config.json")

    def save_edit(self, name: str, value: torch.Tensor, op: str = "save_edit", note: str = ""):
        """Replace tensor `name` by `value` in the overlay and record a history step."""
        self.check_protected(name)
        orig = self.load_original(name)
        value = torch.as_tensor(value).detach().to("cpu", torch.float32).contiguous()
        if tuple(value.shape) != tuple(orig.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(orig.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}: value has NaN or inf")
        ov = dict(self.overlay())
        if torch.equal(value, orig):
            ov.pop(name, None)
        else:
            ov[name] = value
        self._write_overlay(ov)
        step = self._snapshot(op, note=note or name)
        self.journal(op, {"tensor": name, "step": step, "note": note})
        return step

    def drop_edit(self, name: str):
        ov = dict(self.overlay())
        if name not in ov:
            return None
        del ov[name]
        self._write_overlay(ov)
        step = self._snapshot("drop", note=name)
        self.journal("drop", {"tensor": name, "step": step})
        return step

    def _write_overlay(self, ov: dict[str, torch.Tensor]):
        f = self.path / "overlay.safetensors"
        if ov:
            save_file({k: v.contiguous() for k, v in ov.items()}, str(f))
        elif f.exists():
            f.unlink()
        self._overlay = dict(ov)
        if self._model is not None:  # make the in-memory model reflect the new overlay on next use
            params = dict(self._model.named_parameters())
            with torch.no_grad():
                for n, p in params.items():
                    if n in self._base:
                        p.copy_(self._base[n].to(p.device))
            self._loaded = "original"

    # ------------------------------------------------------------------ history / checkpoints
    def _hist(self) -> Path:
        return self.path / "history"

    def steps(self) -> list[dict]:
        out = []
        if self._hist().exists():
            for d in sorted(self._hist().iterdir()):
                if (d / "meta.json").exists():
                    out.append(json.loads((d / "meta.json").read_text()))
        return out

    def _snapshot(self, op: str, tag: str = "", note: str = "") -> int:
        steps = self.steps()
        step = steps[-1]["step"] + 1 if steps else 0
        d = self._hist() / f"{step:04d}"
        d.mkdir(parents=True)
        f = self.path / "overlay.safetensors"
        if f.exists():
            shutil.copy2(f, d / "overlay.safetensors")
        meta = {"step": step, "op": op, "tag": tag, "note": note, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tensors": self.edited()}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return step

    def checkpoint(self, tag: str, note: str = "") -> int:
        """Label the current state. Tags must be unique; a checkpoint is just a named history step."""
        if any(s["tag"] == tag for s in self.steps()):
            raise ValueError(f"tag {tag!r} already used")
        step = self._snapshot("checkpoint", tag=tag, note=note)
        self.journal("checkpoint", {"tag": tag, "step": step, "note": note})
        return step

    def resolve(self, target: str) -> dict:
        steps = self.steps()
        for s in steps:
            if s["tag"] == target:
                return s
        if target.lstrip("-").isdigit():
            k = int(target)
            if k < 0:  # -1 = the step before the current one
                return steps[k - 1]
            for s in steps:
                if s["step"] == k:
                    return s
        raise KeyError(f"no step or tag {target!r}; see `wsp history`")

    def rollback(self, target: str) -> int:
        s = self.resolve(target)
        src = self._hist() / f"{s['step']:04d}" / "overlay.safetensors"
        ov = load_file(src) if src.exists() else {}
        self._write_overlay(ov)
        step = self._snapshot("rollback", note=f"to step {s['step']} {s['tag']}".strip())
        self.journal("rollback", {"to": s["step"], "tag": s["tag"], "step": step})
        return step

    def reset(self) -> int:
        return self.rollback("base")

    # ------------------------------------------------------------------ budget / journal
    def budget_used(self) -> int:
        f = self.path / "budget.json"
        return json.loads(f.read_text())["used"] if f.exists() else 0

    def spend(self, n: int = 1, what: str = ""):
        limit = self.config.get("budget")
        used = self.budget_used()
        if limit is not None and used + n > limit:
            raise BudgetExceeded(f"forward-pass budget exhausted: {used}/{limit} used")
        (self.path / "budget.json").write_text(json.dumps({"used": used + n}))
        self.journal("spend", {"n": n, "what": what, "used": used + n, "limit": limit})

    def journal(self, op: str, data: dict):
        with open(self.path / "journal.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "op": op, **data}, default=str) + "\n")

    def next_dir(self, kind: str) -> Path:
        d = self.path / kind
        d.mkdir(exist_ok=True)
        k = len(list(d.iterdir()))
        return d / f"{k:04d}"
