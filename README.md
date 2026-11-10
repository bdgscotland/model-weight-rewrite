# model-weight-rewrite

An experiment in **weight-space programming**: an LLM (Claude, driven through Claude Code) edits the weights of a
small language model **by hand**. It inspects and changes tensors with tools, with no gradient training. The work is
modelled on MIT's ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(Nov 2026).

- `docs/research.md`: what that work did and released, and what we verified.
- `docs/experiment-design.md`: targets, tasks, metrics, budgets, the experiment ladder, baseline numbers, open
  questions.
- `src/wsp/`: the harness (`wsp` CLI and a Python library).
- `data/tasks/`: small evaluation sets.
- `scripts/`: importing the paper's released edits, and a small Qwen loop check.

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
