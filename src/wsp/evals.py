"""Evaluation tasks. Each takes (ws, split, which) and returns a flat dict of metrics.

split "dev" is for the editor to use while working; "test" is held out for the experimenter (do not tune on it).
All tasks are deterministic (fixed data, fixed seeds, greedy decoding)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .probe import generate
from .workspace import ROOT

TASKS_DIR = ROOT / "data" / "tasks"
CACHE = ROOT / "data" / "cache"


# ---------------------------------------------------------------- perplexity (collateral damage)
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
            "b08601e04326c79dfdd32d625aee71d232d685c3")  # dataset repo revision, pinned


def wikitext_text() -> str:
    f = CACHE / "wikitext2_test.txt"
    if not f.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(WIKITEXT[0], WIKITEXT[1], repo_type="dataset", revision=WIKITEXT[2],
                            cache_dir=CACHE / "hf")
        text = "".join(pq.read_table(p).column("text").to_pylist())
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return f.read_text()


@torch.inference_mode()
def task_ppl(ws, split="dev", which="edited", n_tokens=16384, window=512, batch=8):
    text = wikitext_text()
    half = len(text) // 2
    text = text[:half] if split == "dev" else text[half:]
    ids = ws.tok(text[: n_tokens * 8])["input_ids"][:n_tokens]
    chunks = torch.tensor(ids[: len(ids) // window * window]).view(-1, window)
    model = ws.model(which)
    nll, n = 0.0, 0
    for b in range(0, len(chunks), batch):
        x = chunks[b:b + batch].to(ws.device)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        nll -= lp.sum().item()
        n += lp.numel()
    return {"ppl": math.exp(nll / n), "nll": nll / n, "tokens": n}


# ---------------------------------------------------------------- induction (copying repeated random tokens)
@torch.inference_mode()
def task_induction(ws, split="dev", which="edited", n_seq=64, half=48):
    g = torch.Generator().manual_seed(0 if split == "dev" else 1)
    vocab = min(ws.tok.vocab_size, 50000)
    rand = torch.randint(1000, vocab, (n_seq, half), generator=g)
    bos = ws.tok.bos_token_id if ws.tok.bos_token_id is not None else ws.tok.eos_token_id
    x = torch.cat([torch.full((n_seq, 1), bos), rand, rand], 1).to(ws.device)
    logits = ws.model(which)(input_ids=x).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1).cpu()
    first, second = -lp[:, 1:half].mean().item(), -lp[:, half + 1:].mean().item()
    acc2 = (logits[:, half:-1].argmax(-1).cpu() == x[:, half + 1:].cpu()).float().mean().item()
    return {"loss_first_half": first, "loss_second_half": second, "induction_gain": first - second,
            "second_half_top1": acc2}


# ---------------------------------------------------------------- next-token facts
def _target_scores(ws, prompts, targets, which):
    """Teacher-forced log-prob of each target and whether its first token is the argmax."""
    from .probe import forward

    res, _ = forward(ws, prompts, which=which, topk=1, targets=targets)
    return res


def task_facts(ws, split="dev", which="edited", file="facts.jsonl"):
    rows = [json.loads(l) for l in (TASKS_DIR / file).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    res = _target_scores(ws, [r["prompt"] for r in rows], [r["target"] for r in rows], which)
    top1 = [r["target_first_token_rank"] == 0 for r in res]
    return {"top1_acc": sum(top1) / len(top1), "mean_target_logprob": sum(r["target_logprob"] for r in res) / len(res),
            "n": len(res)}


# ---------------------------------------------------------------- counterfactual fact edit (CounterFact-style)
def task_counterfact(ws, split="dev", which="edited", file="edit_eiffel_rome.json"):
    """Metrics as in ROME's CounterFact: for prompts about the edited subject, success = P(new) > P(true);
    for neighbourhood prompts (other subjects), success = P(true answer) > P(new) is preserved."""
    spec = json.loads((TASKS_DIR / file).read_text())
    new, old = spec["target_new"], spec["target_true"]
    out = {}
    for group in ("rewrite", "paraphrase", "neighborhood"):
        items = [p for p in spec[group] if p.get("split", "dev") == split or group == "rewrite"]
        if not items:
            continue
        prompts = [p["prompt"] for p in items]
        a = _target_scores(ws, prompts, [new] * len(prompts), which)
        if group == "neighborhood":
            b = _target_scores(ws, prompts, [p["true"] for p in items], which)
            ok = [y["target_logprob"] > x["target_logprob"] for x, y in zip(a, b)]
        else:
            b = _target_scores(ws, prompts, [old] * len(prompts), which)
            ok = [x["target_logprob"] > y["target_logprob"] for x, y in zip(a, b)]
        out[f"{group}_success"] = sum(ok) / len(ok)
        out[f"{group}_mean_logp_new"] = sum(x["target_logprob"] for x in a) / len(a)
    return out


# ---------------------------------------------------------------- repetition under greedy decoding (loop analogue)
def rep_n(ids: list[int], n: int = 4) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def task_repetition(ws, split="dev", which="edited", max_new_tokens=128, file="repetition_prompts.json"):
    prompts = json.loads((TASKS_DIR / file).read_text())[split]
    outs = generate(ws, prompts, which=which, max_new_tokens=max_new_tokens, greedy=True)
    eos = ws.tok.eos_token_id
    reps, lens = [], []
    for o in outs:
        if eos in o:
            o = o[: o.index(eos)]
        reps.append(rep_n(o, 4))
        lens.append(len(o))
    return {"rep4": sum(reps) / len(reps), "loop_share": sum(r > 0.5 for r in reps) / len(reps),
            "mean_len": sum(lens) / len(lens), "n": len(prompts),
            "example": ws.tok.decode(outs[0][:60])}


TASKS = {"ppl": task_ppl, "induction": task_induction, "facts": task_facts,
         "counterfact": task_counterfact, "repetition": task_repetition}
DEFAULT_TASKS = ["ppl", "induction", "facts", "counterfact", "repetition"]
