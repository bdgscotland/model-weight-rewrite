# Predicting stalls and wedges in autonomous agent runs

This guide is for applying the repo to its intended problem: given the logs of many runs of an autonomous coding or
building agent, predict where a run is about to **stall** (stop making progress) or **wedge** (get stuck repeating
itself), before it happens.

The agent itself may run on a hosted model whose weights are not available. The approach here does not need them:
a small **open-weights model reads the run transcript**, and the prediction is read from that model's internal
state. Optionally the predictor is then written into the open model's weights, so the model itself outputs the
warning.

## The technique in one paragraph

When a model reads text, its internal activations take different values in different situations. Show it many
transcript excerpts that were followed by a stall and many that were not, record the activations for each, and compute
the single direction that best separates the two groups (the difference of the group averages, corrected for how the
activations normally vary). Projecting a new excerpt onto that direction gives one score: high means "a stall is
likely". This is a closed-form calculation with no gradient training. `wsp detect` implements it; the same
mechanism was used to stop repetition loops and to rewrite a fact in GPT-2 (see `docs/results/`).

## What you need

- Run logs for a few hundred runs or more, with enough per-step text to reconstruct what the agent did.
- For each run, the step where the stall or wedge began, or a rule that determines it (for example: no new commit,
  passing test or file change for N steps; the same tool call repeated K times; a human intervened).
- An open model that fits on the machine. GPT-2 small (124M, 1,024-token context) is enough to try the pipeline.
  Agent transcripts are long, so a model with a larger context is usually needed; Qwen3.5-0.8B is supported by the
  harness. Each extra model is a `wsp download` away.

## Steps

```bash
uv sync
uv run wsp download openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
uv run wsp init runs/reader --model models/openai-community--gpt2 --protect 'wte|wpe|ln_f'
```

**1. Put the runs in one JSONL file**, one run per line:

```json
{"run_id": "r17", "steps": ["text of step 0", "text of step 1", "..."], "event_step": 42}
```

`event_step` is the step where the stall begins, or `null` for a run that finished normally. Each step's text is
whatever you want the reader model to see: the agent's message, the tool call and a trimmed result, log lines.

**2. Make labelled records.** Each record is the last few steps up to some point, labelled by whether the stall begins
within the next `--horizon` steps. Steps at or after the stall are dropped, so the task is to see it coming.

```bash
uv run python scripts/make_records.py runs.jsonl records.jsonl --horizon 10 --context 8 --every 5
```

**3. Compare predictors** on splits that keep each run entirely on one side:

```bash
uv run python scripts/predict_events.py runs/reader records.jsonl --out results.json
```

This prints held-out AUC (0.5 is chance, 1.0 is perfect) for three kinds of predictor: plain text statistics with no
model, the closed-form detector at several layers, and a gradient-trained probe on the same activations.

**4. Fit the chosen detector and confirm it on runs it has never seen.** Set aside some runs before step 3 (for
example the most recent ones) and score them once at the end.

```bash
uv run wsp detect runs/reader fit stall --data records_fit.jsonl --point hidden:9 --pool mean
uv run wsp detect runs/reader score stall --data records_new.jsonl --out scores.jsonl
```

**5. Optional: write the detector into the weights**, so the model itself answers a question about the transcript.
This needs a detector fitted on an MLP's neurons at the last token, with background positions:

```bash
Q=$'\nWill this run stall soon? Answer:'
uv run wsp detect runs/reader fit stall_w --data records_fit.jsonl --suffix "$Q" \
    --point transformer.h.11.mlp.c_proj:in --pool last --background 6000
uv run wsp detect runs/reader write stall_w --pos-token " yes" --neg-token " no" --alpha 40
uv run wsp detect runs/reader eval stall_w --data records_new.jsonl --pos-token " yes" --neg-token " no"
uv run wsp eval runs/reader --task ppl --task facts        # what the edit cost the model elsewhere
uv run wsp save-model out/stall-predictor                  # a normal Hugging Face checkpoint
```

For prediction alone, step 4 is enough. Step 5 matters only if the predictor has to live inside a model.

## What the experiments here say to expect

| Test | Result | Where |
|---|---|---|
| Classify movie-review sentiment (a property of text already read) | Detector AUC 0.92, level with a trained probe; written into the weights, accuracy 0.81 at +1.3% perplexity | `docs/results/detector-demo/` |
| Predict that a small model's run will end in a loop, from its first 0-512 tokens | Weak: AUC 0.5-0.7, barely ahead of plain transcript statistics | `docs/results/stall-prediction/` |

So the tooling works, the closed-form detector is as good as a trained probe on the same activations, and predicting
something that has not happened yet is much harder than classifying what is on the page. None of this has been run
on real agent logs. Treat the text-statistics baseline as the number to beat: if it already predicts stalls well, the
language model adds little.

## Pitfalls

- **Leakage across steps of one run.** Neighbouring steps are near-duplicates. Always split by run (`group`), never
  by record. The tools here do this.
- **Recognising instead of predicting.** If records include steps after the stall began, the predictor only learns to
  spot a stall in progress. `make_records.py` drops them; check that `event_step` marks the beginning, not the moment
  someone noticed.
- **Label quality.** A rule such as "no progress for N steps" places the stall N steps late. Shift it back, or the
  last N steps before the label are already stalled and the task looks easier than it is.
- **Picking the best of many settings.** With a few hundred runs, the best of 16 layer and pooling combinations will
  look better than it is. Confirm the chosen setting once on untouched runs.
- **Class imbalance.** Stalls are rare per step. Read AUC, recall and false-alarm rate, not accuracy.
- **Context length.** Texts longer than the model's context keep only their most recent tokens. Tune `--context` so a
  record fits, or use a model with a longer context.
- **Run-level shortcuts.** The predictor may learn which task or repository a run belongs to instead of how it is
  going. If stalls cluster in a few tasks, also split by task.
- **Averaging beats the last token.** In the stand-in, averaging activations over the excerpt (`--pool mean`) was
  consistently better than reading only the final token. Writing into the weights needs `--pool last`.

## If the prediction is weak

Three checks, cheapest first: (1) see how well a person can tell from the same excerpt; (2) lengthen `--context` and
shorten `--horizon`, since a stall is easier to see just before it starts; (3) add structured features from the logs
to `text_stats` in `scripts/predict_events.py` (tool-call repetition, time since the last change that stuck, error
counts), which may carry more signal than the prose.
