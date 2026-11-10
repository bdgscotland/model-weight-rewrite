"""Forward passes: next-token predictions, target log-probs, greedy generation, and activation capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _encode(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}


@torch.inference_mode()
def forward(ws, texts: list[str], which="edited", topk=10, targets: list[str] | None = None,
            capture: list[str] = (), hidden: bool = False, attn: bool = False, out_dir: Path | None = None):
    """Run the model on `texts` (right-padded). Returns per-text summaries; saves captured tensors to out_dir.

    capture: module names (see `wsp modules`); each module's output is saved as [batch, seq, ...].
             Append ":in" to save the module's input instead (e.g. "model.layers.15.mlp.down_proj:in" = MLP neurons).
    hidden:  residual stream after each block (index 0 = embeddings), saved as hidden_states.npy [L+1, B, T, D].
    attn:    attention probabilities, saved as attentions.npy [L, B, H, T, T].
    targets: per text, a continuation whose log-prob is computed teacher-forced (e.g. " Paris")."""
    model, tok = ws.model(which), ws.tok
    tok.padding_side = "right"
    full = [t + (targets[i] if targets else "") for i, t in enumerate(texts)]
    enc = _encode(tok, full, ws.device)
    prompt_lens = [len(tok(t)["input_ids"]) for t in texts]
    seq_lens = enc["attention_mask"].sum(1).tolist()

    grabbed, hooks = {}, []
    mods = dict(model.named_modules())
    for name in capture:
        mod_name, want_input = (name[:-3], True) if name.endswith(":in") else (name, False)
        if mod_name not in mods:
            raise KeyError(f"no module {mod_name!r}; see `wsp modules`")

        def hook(_m, i, o, name=name, want_input=want_input):
            o = i[0] if want_input else (o[0] if isinstance(o, tuple) else o)
            grabbed[name] = o.detach().float().cpu().numpy()

        hooks.append(mods[mod_name].register_forward_hook(hook))
    try:
        out = model(**enc, output_hidden_states=hidden, output_attentions=attn)
    finally:
        for h in hooks:
            h.remove()

    logp = torch.log_softmax(out.logits.float(), -1).cpu()
    ids = enc["input_ids"].cpu()
    results = []
    for i, text in enumerate(texts):
        pl = prompt_lens[i]
        last = logp[i, pl - 1]
        top = last.topk(topk)
        r = {"text": text, "n_tokens": pl,
             "top": " | ".join(f"{tok.decode([int(j)])!r} {float(v.exp()):.4f}" for v, j in zip(top.values, top.indices))}
        if targets:
            tgt_ids = ids[i, pl:seq_lens[i]]
            lp = logp[i, torch.arange(pl - 1, seq_lens[i] - 1), tgt_ids]
            first = int(tgt_ids[0]) if len(tgt_ids) else None
            r["target"] = targets[i]
            r["target_logprob"] = round(float(lp.sum()), 4)
            r["target_first_token_rank"] = int((last > last[first]).sum()) if first is not None else None
        results.append(r)

    saved = {}
    if out_dir is not None and (grabbed or hidden or attn):
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in grabbed.items():
            np.save(out_dir / f"{name.replace(':in', '.input')}.npy", arr)
            saved[name] = list(arr.shape)
        if hidden:
            arr = torch.stack(out.hidden_states).float().cpu().numpy()
            np.save(out_dir / "hidden_states.npy", arr)
            saved["hidden_states"] = list(arr.shape)
        if attn:
            arr = torch.stack(out.attentions).float().cpu().numpy()
            np.save(out_dir / "attentions.npy", arr)
            saved["attentions"] = list(arr.shape)
        meta = {"texts": texts, "targets": targets, "which": which, "seq_lens": seq_lens,
                "prompt_lens": prompt_lens, "tokens": [tok.convert_ids_to_tokens(ids[i, :seq_lens[i]]) for i in range(len(texts))],
                "files": saved, "padding": "right; positions >= seq_len are padding"}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return results, saved


@torch.inference_mode()
def generate(ws, prompts: list[str], which="edited", max_new_tokens=64, greedy=True, seed=0, batch=32):
    model, tok = ws.model(which), ws.tok
    tok.padding_side = "left"
    outs = []
    for b in range(0, len(prompts), batch):
        chunk = prompts[b:b + batch]
        enc = _encode(tok, chunk, ws.device)
        torch.manual_seed(seed)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=not greedy,
                           pad_token_id=tok.pad_token_id, top_k=None if greedy else 50,
                           top_p=None, temperature=None if greedy else 1.0)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append(row.cpu().tolist())
    tok.padding_side = "right"
    return outs
