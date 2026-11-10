# model-weight-rewrite

Tools and experiments for **editing a small language model's weights by hand**: inspecting and changing tensors
directly, with no gradient training. The editor is an LLM agent (Claude, driven through Claude Code) working through
a command-line harness. The work is modelled on MIT's
["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) (Nov 2026).

The repo has two parts:

1. **A harness (`wsp`)** to look at weights, run the model and record its internal activations, apply edits that
   can always be undone, and measure the effect on held-out tests.
2. **A detector technique built on it**: find the direction in a model's activations that separates two situations,
   and optionally write it into the weights so the model reacts to it. The intended application is **predicting
   stalls and wedges in autonomous agent runs** from their transcripts, using an open model as the reader.
   Start with `docs/stall-prediction-guide.md`.

## What has been shown so far

All on GPT-2 small unless noted; held-out test numbers. Details and scripts are in `docs/results/`.

| Experiment | Hand edit | Result | Standard method on the same task |
|---|---|---|---|
| E1: switch off in-context copying | 5 attention heads replaced by their average output (0.2% of weights) | copying accuracy 0.97 -> 0.01; perplexity +13.5% | n/a |
| E2: rewrite one fact (Eiffel Tower -> Rome) | one rank-1 edit to an MLP | rewrite 1.0, paraphrases 0.67, perplexity +0.3% | ROME: paraphrases 1.0, perplexity +0.01% |
| E3: stop greedy repetition loops | a loop detector wired to a correction, one rank-1 edit | looping prompts 87.5% -> 25%, perplexity +3.0% | repetition penalty at decode time: 0% loops, no weight change |
| Detector demo (sentiment) | closed-form detector, then written into the weights | AUC 0.92 as a probe; accuracy 0.81 written in | trained probe: AUC 0.91 |
| Stall prediction stand-in (Qwen3.5-0.8B runs) | closed-form detector on the first 0-512 tokens | weak: AUC 0.5-0.7 | text statistics: 0.5-0.64 |

In short: hand editing works and is cheap, existing methods beat it where they exist, the closed-form detector
matches a trained probe, and early prediction of stalls is unproven.

## Layout

- `docs/stall-prediction-guide.md`: applying the detector to agent run logs, step by step.
- `docs/research.md`: what the MIT work did and released, and what was verified.
- `docs/experiment-design.md`: targets, tasks, metrics, budgets, the experiment ladder, the protocol for the editor.
- `docs/results/`: one folder per experiment, with the write-up and the scripts used.
- `src/wsp/`: the harness (`wsp` CLI and a Python library).
- `data/tasks/`: small evaluation sets.
- `scripts/`: baselines (`rome_baseline.py`, `decode_baseline.py`), detector demo, stall stand-in
  (`stall_runs.py`, `stall_predict.py`), and the general tools for agent logs (`make_records.py`,
  `predict_events.py`).

## Starting a session

```bash
uv sync                                   # Python 3.12 venv with torch, transformers (MPS on Apple silicon)
uv run --group dev pytest -q              # 5 offline tests (tiny random GPT-2; no download)

# models live in models/ (gitignored); revisions are pinned and recorded in models/*/wsp_source.json
uv run wsp download openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
uv run wsp download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17   # 1.6 GB, optional

# one workspace per experiment attempt; runs/ is gitignored
uv run wsp init runs/e1-induction --model models/openai-community--gpt2 --budget 25 --protect 'wte|wpe|ln_f'
uv run wsp eval runs/e1-induction --model original      # dev baseline (costs 1 pass)
```

Before handing control to the editing agent, give it `docs/experiment-design.md` §4 (the protocol) and the experiment
card for the chosen experiment.

## The tool surface

Every command takes the run directory first. Use `uv run wsp <cmd> -h` for the options.

| Purpose | Commands |
|---|---|
| Look at the model | `info [--filter RE]`, `modules [--filter RE]` (names you can capture), `stats RE [--norms]`, `show NAME --index "[0:4, {3,7}]"`, `svd NAME -k 10 [--save DIR]`, `export NAME out.npy [--original]` |
| Run it (costs 1 pass) | `run --prompt ... [--target " Paris"] [--model original\|edited\|both] [--capture MODULE] [--hidden] [--attn] [--generate N]`. Captures go to `runs/<run>/probes/NNNN/*.npy` with `meta.json` (tokens, lengths). |
| Edit (recorded, reversible) | `edit zero NAME --index ...`, `edit scale NAME --index ... --factor f`, `edit rank1 NAME --u u.npy --v v.npy --alpha a`, `edit project-out NAME --dir d.npy --axis 0\|1 [--strength s] [--only "{cols}"]`, `edit add NAME --npy delta.npy`, `edit set NAME --npy new.npy`, `drop NAME` |
| Check an edit | `diff` (entries changed, relative change, delta rank, % of model), `eval [--task T] [--split dev\|test] [--model original\|edited]`. Tasks: `ppl induction facts counterfact repetition`. A dev eval costs 1 pass. |
| Undo | `history`, `checkpoint TAG`, `rollback TAG\|STEP\|-1`, `reset` (nothing is ever deleted; a rollback is itself undoable) |
| Bookkeeping | `budget`, `save-model OUT_DIR` (a plain HF checkpoint with the edit merged) |

For custom analysis, use the library from a script (`uv run python my_analysis.py`):

```python
from wsp import Workspace
ws = Workspace.open("runs/e1-induction")
W = ws.load("transformer.h.5.attn.c_proj.weight")   # current value (float32 CPU copy)
W0 = ws.load_original("transformer.h.5.attn.c_proj.weight")
W[64:128] = 0                                       # e.g. silence head 5.1's output
ws.save_edit("transformer.h.5.attn.c_proj.weight", W)  # recorded as a history step; respects `protect`
```

Orientation matters. `nn.Linear` weights are `(out, in)` (Qwen). GPT-2's `Conv1D` weights are `(in, out)`, so the
rows of `attn.c_proj.weight` are the concatenated head outputs (head h = rows `64h:64h+64`).

## Detectors (`wsp detect`)

Fit a closed-form direction that separates two kinds of text in the model's activations, check it on held-out
groups, and optionally write it into the weights so the model itself emits the prediction. No gradients.

Data is JSONL, one record per line: `{"text": "...", "label": 0 or 1, "group": "run-17"}`. Records with the same
`group` never straddle the fit/held-out split, so use one group per run. Texts longer than the model's context keep
their last tokens.

```bash
uv run wsp detect RUN fit NAME --data fit.jsonl --point hidden:9             # or a module, e.g. transformer.h.11.mlp.c_proj:in
uv run wsp detect RUN score NAME --data new.jsonl [--out scores.jsonl]       # AUC, accuracy, recall, false-alarm rate
uv run wsp detect RUN write NAME --pos-token " yes" --neg-token " no" --alpha 40   # needs a '<module>:in' point and --background at fit
uv run wsp detect RUN eval NAME --data new.jsonl --pos-token " yes" --neg-token " no"   # original vs edited model
```

`--suffix` appends a fixed question to every record before the model reads it. `--background N` keeps the detector
silent at other token positions, which is required before `write`. A worked example with numbers is in
`docs/results/detector-demo/`.

## How a run directory works

`runs/<name>/` holds `config.json` (model, budget, protected tensors), `overlay.safetensors` (the full values of
edited tensors only), `history/NNNN/` (an append-only snapshot of the overlay at every change), `journal.jsonl` (every
action), `probes/` and `evals/`. **The base model files are never written.** Every state the overlay has ever been in
can be restored with `rollback`.

## Rules for the editing agent (summary; full protocol in docs/experiment-design.md §4)

No gradients, optimizers or training loops. Run the model only through `wsp run` / `wsp eval` so passes are counted.
Choose edits on `--split dev` and leave `--split test` to the experimenter. Do not edit protected tensors. Do not edit
by scaling whole tensors. Report the number of weights changed and the evidence for each edit.

## Repo hygiene

`models/`, `runs/`, `data/cache/` and `external/` are gitignored. `external/lm_understand-logs/` is a sparse clone of
the authors' logs repo, kept **for reading only** (untrusted; none of its code is executed).
`[tool.uv] exclude-newer` in `pyproject.toml` keeps dependency resolution off packages published in the last two
weeks.
