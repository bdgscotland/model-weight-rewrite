"""ROME baseline for the fact-rewrite experiment (E2). This is a GRADIENT method: it is the comparison for the
hand edits, not something the editing agent may use.

Minimal re-implementation of Meng et al. 2022 ("Locating and Editing Factual Associations in GPT") for GPT-2:
  1. key k   = MLP neuron pattern at the subject's last token in layer L, averaged over context prefixes
  2. value   = a vector delta added to that MLP's output, optimised by Adam so the model predicts the new target
               (plus a KL term that keeps "<subject> is a" unchanged, and weight decay)
  3. update  = rank-1:  W += (C^-1 k / k.C^-1 k) delta^T,  C = second moment of keys on ordinary text

Differences from the paper: C is estimated on text sampled from the model itself (about 20k tokens), not on 100k
Wikipedia samples; context prefixes are a fixed list, not sampled.

    uv run python scripts/rome_baseline.py RUN_DIR [--layer 5] [--steps 25]

RUN_DIR is a fresh `wsp init` workspace; the edit is saved into it, so `wsp eval` then scores it as usual.
"""
import argparse
import json
from pathlib import Path

import torch

from wsp import Workspace
from wsp.probe import generate

PREFIXES = ["", "Yesterday I read something interesting.", "Here is a fact.", "My friend told me this.",
            "People often forget this.", "As everyone knows,", "It is worth remembering that", "In case you wondered,",
            "A short note for travellers.", "The guide book says:"]
SEEDS = ["The", "In", "A", "I", "He", "She", "It", "We", "They", "This", "There", "When", "After", "On", "One", "My",
         "Some", "Many", "At", "As", "For", "But", "What", "How", "New", "Last", "People", "According", "During",
         "If", "To", "With"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--kl", type=float, default=0.0625)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=4.0)
    ap.add_argument("--cov-batches", type=int, default=5)
    ap.add_argument("--task", default="data/tasks/edit_eiffel_rome.json")
    a = ap.parse_args()

    ws = Workspace.open(a.run)
    spec = json.loads(Path(a.task).read_text())
    subject, new = spec["subject"], spec["target_new"]
    request = spec["rewrite"][0]["prompt"]                      # the edit request itself, not an evaluation paraphrase
    assert subject in request
    tok, dev, model = ws.tok, ws.device, ws.model("original")
    mlp = model.transformer.h[a.layer].mlp
    new_id = tok(new)["input_ids"]
    assert len(new_id) == 1, "single-token targets only"
    new_id = new_id[0]

    def encode(texts):
        ids = [tok(t)["input_ids"] for t in texts]
        n = max(map(len, ids))
        x = torch.full((len(ids), n), tok.eos_token_id)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        return x.to(dev), [len(s) for s in ids]

    def with_prefix(text):
        return [(p + " " + text).strip() for p in PREFIXES]

    def subj_pos(texts):                                        # index of the subject's last token in each text
        return [len(tok(t[: t.index(subject) + len(subject)])["input_ids"]) - 1 for t in texts]

    grabbed = {}
    h_in = mlp.c_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("z", i[0].detach()))
    h_out = mlp.register_forward_hook(lambda m, i, o: grabbed.__setitem__("out", o.detach()))

    # 1. key
    texts = with_prefix(request)
    x, lens = encode(texts)
    pos = subj_pos(texts)
    rows = torch.arange(len(texts))
    with torch.no_grad():
        model(input_ids=x)
    k = grabbed["z"][rows, pos].mean(0).cpu().double()
    v0_norm = grabbed["out"][rows, pos].norm(dim=-1).mean().item()

    # 3a. key second moment on text sampled from the model
    zs = []
    for s in range(a.cov_batches):
        outs = generate(ws, SEEDS, which="original", max_new_tokens=128, greedy=False, seed=100 + s)
        xs, ls = encode([p + tok.decode(o) for p, o in zip(SEEDS, outs)])
        with torch.no_grad():
            model(input_ids=xs)
        zs += [grabbed["z"][i, : ls[i]].cpu().double() for i in range(len(ls))]
    Z = torch.cat(zs)
    C = Z.T @ Z / len(Z)
    h_in.remove(); h_out.remove()

    # 2. value: optimise delta added to the MLP output at the subject's last token
    kl_texts = [f"The {subject} is a"]
    xk, lk = encode(kl_texts)
    pk = subj_pos(kl_texts)
    with torch.no_grad():
        base_kl = torch.log_softmax(model(input_ids=xk).logits[0, lk[0] - 1], -1)
    delta = torch.zeros(model.config.n_embd, device=dev, requires_grad=True)
    cur = {}

    def add_delta(m, i, o):
        o = o.clone()
        o[torch.arange(len(cur["pos"])), cur["pos"]] += delta
        return o

    hook = mlp.register_forward_hook(add_delta)
    opt = torch.optim.Adam([delta], lr=a.lr)
    last = torch.tensor(lens) - 1
    for step in range(a.steps):
        opt.zero_grad()
        cur["pos"] = pos
        lp = torch.log_softmax(model(input_ids=x).logits[rows, last], -1)[:, new_id]
        cur["pos"] = pk
        kl_now = torch.log_softmax(model(input_ids=xk).logits[0, lk[0] - 1], -1)
        kl = torch.nn.functional.kl_div(kl_now, base_kl, log_target=True, reduction="sum")
        loss = -lp.mean() + a.kl * kl + a.decay * (delta.norm() / v0_norm) ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            if delta.norm() > a.clamp * v0_norm:
                delta.mul_(a.clamp * v0_norm / delta.norm())
        if step % 5 == 0 or step == a.steps - 1:
            print(f"step {step:2d}  loss {loss.item():.3f}  p(new) {lp.exp().mean().item():.3f}  kl {kl.item():.4f}  "
                  f"|delta| {delta.norm().item():.1f}")
    hook.remove()

    # 3b. rank-1 update
    u = torch.linalg.solve(C, k)
    u = u / (u @ k)
    name = f"transformer.h.{a.layer}.mlp.c_proj.weight"
    W = ws.load(name)
    ws.save_edit(name, W + torch.outer(u.float(), delta.detach().cpu().float()), op="rome_baseline",
                 note=f"layer {a.layer}, {a.steps} Adam steps, {len(Z)} keys for C")
    print(f"saved rank-1 ROME edit to {name}: |delta| {delta.norm().item():.1f}, {len(Z)} keys in C")


if __name__ == "__main__":
    main()
