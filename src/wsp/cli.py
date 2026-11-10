"""wsp: command-line tool surface for hand-editing a small language model's weights.

Every command takes the run directory first (e.g. runs/gpt2-e1). Run `wsp <command> -h` for details."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import tensors as T
from .workspace import ROOT, Workspace


def _p(obj):
    print(json.dumps(obj, indent=1, default=str) if not isinstance(obj, str) else obj)


def _match(ws, pattern):
    names = [n for n in ws.names() if re.search(pattern, n)]
    if not names:
        sys.exit(f"no tensor matches {pattern!r}; see `wsp info`")
    return names


# ---------------------------------------------------------------- setup
def cmd_download(a):
    from huggingface_hub import snapshot_download

    out = ROOT / "models" / a.model_id.replace("/", "--")
    path = snapshot_download(a.model_id, revision=a.revision, local_dir=out,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
                             ignore_patterns=["onnx/*", "*/*"])  # top-level files only
    from huggingface_hub import HfApi

    sha = HfApi().model_info(a.model_id, revision=a.revision).sha
    (Path(path) / "wsp_source.json").write_text(json.dumps({"model_id": a.model_id, "revision": sha}, indent=1))
    _p({"model_dir": str(Path(path).relative_to(ROOT)), "model_id": a.model_id, "revision": sha})


def cmd_init(a):
    ws = Workspace.create(a.run, a.model, device=a.device, budget=a.budget, protect=a.protect or [], note=a.note)
    _p({"run": str(ws.path), **ws.config})


# ---------------------------------------------------------------- inspection
def cmd_info(a):
    ws = Workspace.open(a.run)
    m = ws.model("original")
    mods = dict(m.named_modules())
    total = sum(t.numel() for t in ws.base().values())
    rows = []
    for n, t in ws.base().items():
        if a.filter and not re.search(a.filter, n):
            continue
        owner = mods.get(n.rsplit(".", 1)[0])
        rows.append(f"{n:60s} {str(tuple(t.shape)):18s} {type(owner).__name__ if owner else '':12s}"
                    f"{' EDITED' if n in ws.overlay() else ''}")
    print(f"model: {ws.config['model_dir']}  class: {type(m).__name__}  params: {total:,}  device: {ws.device}")
    print(f"config: layers={getattr(m.config, 'num_hidden_layers', getattr(m.config, 'n_layer', '?'))} "
          f"d_model={getattr(m.config, 'hidden_size', getattr(m.config, 'n_embd', '?'))} vocab={m.config.vocab_size}")
    print("note: nn.Linear stores weight as (out, in); GPT-2's Conv1D stores (in, out).")
    print("\n".join(rows))
    print(f"edited tensors: {ws.edited() or 'none'}   budget used: {ws.budget_used()}/{ws.config.get('budget')}")


def cmd_modules(a):
    ws = Workspace.open(a.run)
    for n, mod in ws.model("original").named_modules():
        if n and (not a.filter or re.search(a.filter, n)):
            print(f"{n:60s} {type(mod).__name__}")


def cmd_stats(a):
    ws = Workspace.open(a.run)
    for n in _match(ws, a.pattern):
        t = ws.load_original(n) if a.original else ws.load(n)
        s = T.stats(t)
        if a.norms:
            s.update(T.row_col_norms(t, a.k))
        print(n, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()}))


def cmd_show(a):
    ws = Workspace.open(a.run)
    t = ws.load_original(a.name) if a.original else ws.load(a.name)
    torch.set_printoptions(precision=a.precision, linewidth=160, sci_mode=False, threshold=a.max_items)
    sl = t[T.parse_index(a.index)]
    print(f"{a.name}{a.index or ''} shape={tuple(sl.shape)}")
    print(sl)


def cmd_svd(a):
    ws = Workspace.open(a.run)
    t = ws.load_original(a.name) if a.original else ws.load(a.name)
    U, S, Vh = T.svd(t, a.k)
    print(f"{a.name} shape={tuple(t.shape)} top-{a.k} singular values: {[round(x, 4) for x in S[:a.k].tolist()]}")
    print(f"energy in top-{a.k}: {(S[:a.k] ** 2).sum().item() / (S ** 2).sum().item():.4f}")
    if a.save:
        out = Path(a.save)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "U.npy", U.numpy()); np.save(out / "S.npy", S.numpy()); np.save(out / "Vh.npy", Vh.numpy())
        print(f"saved U[:, :k], S, Vh[:k] to {out}")


def cmd_export(a):
    ws = Workspace.open(a.run)
    t = ws.load_original(a.name) if a.original else ws.load(a.name)
    np.save(a.out, t.numpy())
    print(f"saved {a.name} {tuple(t.shape)} -> {a.out}")


def cmd_diff(a):
    ws = Workspace.open(a.run)
    total = sum(t.numel() for t in ws.base().values())
    changed = 0
    for n in ws.edited():
        d = T.delta_summary(ws.load_original(n), ws.load(n))
        changed += d["changed_entries"]
        print(n, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}))
    print(f"total changed weights: {changed:,} of {total:,} ({100 * changed / total:.4f}%)")


# ---------------------------------------------------------------- forward passes
def _texts(a):
    texts = list(a.prompt or [])
    if a.prompts_file:
        texts += [l.rstrip("\n") for l in open(a.prompts_file) if l.strip()]
    if not texts:
        sys.exit("give --prompt or --prompts-file")
    return texts


def cmd_run(a):
    from .probe import forward, generate

    ws = Workspace.open(a.run)
    texts = _texts(a)
    targets = None
    if a.target:
        targets = a.target * len(texts) if len(a.target) == 1 else a.target
        if len(targets) != len(texts):
            sys.exit("give one --target, or one per prompt")
    which = ["original", "edited"] if a.model == "both" else [a.model]
    ws.spend(1, f"run {a.model} n={len(texts)}")
    out = {}
    for w in which:
        res = {}
        if a.generate:
            gens = generate(ws, texts, which=w, max_new_tokens=a.generate, greedy=not a.sample)
            res["generations"] = [ws.tok.decode(g) for g in gens]
        d = ws.next_dir("probes") if (a.capture or a.hidden or a.attn) else None
        r, saved = forward(ws, texts, which=w, topk=a.topk, targets=targets, capture=a.capture or [],
                           hidden=a.hidden, attn=a.attn, out_dir=d)
        res["results"] = r
        if d:
            res["saved_to"] = str(d)
            res["saved"] = saved
        out[w] = res
    ws.journal("run", {"model": a.model, "n": len(texts), "capture": a.capture, "saved": [o.get("saved_to") for o in out.values()]})
    _p(out)


def cmd_eval(a):
    from .evals import DEFAULT_TASKS, TASKS

    ws = Workspace.open(a.run)
    tasks = a.task or DEFAULT_TASKS
    if a.split == "dev":
        ws.spend(1, f"eval dev {','.join(tasks)} {a.model}")
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "split": a.split, "model": a.model,
           "edited_tensors": ws.edited(), "step": ws.steps()[-1]["step"], "results": {}}
    for t in tasks:
        t0 = time.time()
        res["results"][t] = TASKS[t](ws, split=a.split, which=a.model)
        res["results"][t]["seconds"] = round(time.time() - t0, 1)
    f = ws.next_dir("evals").with_suffix(".json")
    f.write_text(json.dumps(res, indent=1))
    ws.journal("eval", {"file": str(f), "split": a.split, "model": a.model,
                        "summary": {t: {k: v for k, v in r.items() if k != "example"} for t, r in res["results"].items()}})
    _p(res)


# ---------------------------------------------------------------- edits
def cmd_edit(a):
    ws = Workspace.open(a.run)
    w = ws.load(a.name)
    idx = T.parse_index(a.index) if getattr(a, "index", None) else None
    if a.op == "set":
        new = T.load_array(a.npy)
    elif a.op == "add":
        new = T.op_add(w, T.load_array(a.npy), a.alpha)
    elif a.op == "scale":
        if idx is None or a.factor is None:
            sys.exit("scale needs --index and --factor (scaling a whole tensor by a constant is a global knob)")
        new = T.op_scale(w, idx, a.factor)
    elif a.op == "zero":
        if idx is None:
            sys.exit("zero needs --index")
        new = T.op_set_value(w, idx, 0.0)
    elif a.op == "rank1":
        new = T.op_rank1(w, T.load_array(a.u), T.load_array(a.v), a.alpha)
    elif a.op == "project-out":
        sel = T.parse_index(a.only)[0] if a.only else None
        new = T.op_project_out(w, T.load_array(a.dir), a.axis, a.strength, sel)
    else:
        sys.exit(a.op)
    summary = T.delta_summary(w, new)
    step = ws.save_edit(a.name, new, op=f"edit:{a.op}", note=a.note or " ".join(sys.argv[2:]))
    _p({"tensor": a.name, "step": step, "delta_vs_previous": summary,
        "delta_vs_original": T.delta_summary(ws.load_original(a.name), ws.load(a.name))})


def cmd_drop(a):
    ws = Workspace.open(a.run)
    _p({"dropped": a.name, "step": ws.drop_edit(a.name)})


def cmd_history(a):
    ws = Workspace.open(a.run)
    for s in ws.steps():
        print(f"{s['step']:4d}  {s['time']}  {s['op']:18s} {s['tag']:14s} tensors={len(s['tensors'])}  {s['note']}")


def cmd_checkpoint(a):
    ws = Workspace.open(a.run)
    _p({"tag": a.tag, "step": ws.checkpoint(a.tag, a.note)})


def cmd_rollback(a):
    ws = Workspace.open(a.run)
    step = ws.rollback(a.target)
    _p({"rolled_back_to": a.target, "new_step": step, "edited_tensors": ws.edited()})


def cmd_reset(a):
    ws = Workspace.open(a.run)
    _p({"reset_step": ws.reset(), "edited_tensors": ws.edited()})


def cmd_budget(a):
    ws = Workspace.open(a.run)
    _p({"used": ws.budget_used(), "limit": ws.config.get("budget")})


def cmd_save_model(a):
    ws = Workspace.open(a.run)
    m = ws.model("edited")
    m.save_pretrained(a.out)
    ws.tok.save_pretrained(a.out)
    (Path(a.out) / "wsp_edit.json").write_text(json.dumps({"run": str(ws.path), "edited": ws.edited(),
                                                           "step": ws.steps()[-1]["step"], **ws.config}, indent=1))
    print(f"saved edited model to {a.out}")


# ---------------------------------------------------------------- parser
def build_parser():
    P = argparse.ArgumentParser(prog="wsp", description=__doc__)
    sub = P.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help, run=True):
        p = sub.add_parser(name, help=help, description=help)
        if run:
            p.add_argument("run", help="run directory, e.g. runs/gpt2-e1")
        p.set_defaults(fn=fn)
        return p

    p = add("download", cmd_download, "download a model from Hugging Face into models/", run=False)
    p.add_argument("model_id"); p.add_argument("--revision", default=None)

    p = add("init", cmd_init, "create a run workspace over a downloaded model")
    p.add_argument("--model", required=True, help="model dir, e.g. models/openai-community--gpt2")
    p.add_argument("--device", default="auto"); p.add_argument("--budget", type=int, default=None,
                   help="max forward passes (each `run` and each dev `eval` costs 1)")
    p.add_argument("--protect", action="append", help="regex of tensor names edits may not touch (repeatable)")
    p.add_argument("--note", default="")

    p = add("info", cmd_info, "list tensors (name, shape, owning module type) and edit status")
    p.add_argument("--filter")
    p = add("modules", cmd_modules, "list module names usable with `run --capture`")
    p.add_argument("--filter")

    p = add("stats", cmd_stats, "statistics of every tensor whose name matches a regex")
    p.add_argument("pattern"); p.add_argument("--original", action="store_true")
    p.add_argument("--norms", action="store_true", help="also largest rows/cols by norm"); p.add_argument("-k", type=int, default=5)

    p = add("show", cmd_show, "print a slice of a tensor")
    p.add_argument("name"); p.add_argument("--index", help='e.g. "[0:4, 0:8]" or "[{3,7}, :10]"')
    p.add_argument("--original", action="store_true"); p.add_argument("--precision", type=int, default=4)
    p.add_argument("--max-items", type=int, default=400)

    p = add("svd", cmd_svd, "top singular values of a 2-D tensor (optionally save U, S, Vh)")
    p.add_argument("name"); p.add_argument("-k", type=int, default=10); p.add_argument("--save")
    p.add_argument("--original", action="store_true")

    p = add("export", cmd_export, "save a tensor (current or original) to .npy for your own analysis")
    p.add_argument("name"); p.add_argument("out"); p.add_argument("--original", action="store_true")

    add("diff", cmd_diff, "what the current edit changes relative to the original weights")

    p = add("run", cmd_run, "forward pass: top-k next tokens, target log-probs, generation, activation capture")
    p.add_argument("--prompt", action="append"); p.add_argument("--prompts-file")
    p.add_argument("--target", action="append", help="continuation to score (one, or one per prompt)")
    p.add_argument("--model", choices=["edited", "original", "both"], default="edited")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--capture", action="append", help="module whose output to save (repeatable); NAME:in saves its input, e.g. ...mlp.c_proj:in = MLP neurons")
    p.add_argument("--hidden", action="store_true", help="save the residual stream after every layer")
    p.add_argument("--attn", action="store_true", help="save attention probabilities")
    p.add_argument("--generate", type=int, default=0, help="also generate this many tokens (greedy)")
    p.add_argument("--sample", action="store_true", help="sample instead of greedy when generating")

    p = add("eval", cmd_eval, "run evaluation tasks; results saved under evals/")
    p.add_argument("--task", action="append", help="ppl, induction, facts, counterfact, repetition (default: all)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--model", choices=["edited", "original"], default="edited")

    p = add("edit", cmd_edit, "apply an edit to one tensor (auto-recorded in history; undo with rollback)")
    p.add_argument("op", choices=["set", "add", "scale", "zero", "rank1", "project-out"])
    p.add_argument("name"); p.add_argument("--index"); p.add_argument("--factor", type=float)
    p.add_argument("--npy"); p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--u"); p.add_argument("--v")
    p.add_argument("--dir"); p.add_argument("--axis", type=int); p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--only", help='project-out: restrict to these columns (axis 0) / rows (axis 1), e.g. "{3,9}"')
    p.add_argument("--note", default="")

    p = add("drop", cmd_drop, "remove one tensor from the edit (back to its original value)")
    p.add_argument("name")
    add("history", cmd_history, "list every step (edits, checkpoints, rollbacks)")
    p = add("checkpoint", cmd_checkpoint, "label the current state")
    p.add_argument("tag"); p.add_argument("--note", default="")
    p = add("rollback", cmd_rollback, "return to a step number, a tag, or -1 (the previous step)")
    p.add_argument("target")
    add("reset", cmd_reset, "return to the unedited model (a new step; nothing is deleted)")
    add("budget", cmd_budget, "forward passes used / limit")
    p = add("save-model", cmd_save_model, "write the edited model as a normal HF checkpoint")
    p.add_argument("out")
    return P


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
