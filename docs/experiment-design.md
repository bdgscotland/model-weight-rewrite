# Proposed experiment: Claude edits small language models by hand

The evidence labels are the same as in `docs/research.md`: [primary], [ran], [secondary], [inference].

## 1. The question

Can Claude, working in Claude Code with a tool surface over the weights and a fixed forward-pass budget, **diagnose a
behaviour in a small LM and change it with targeted weight edits**? The edits must be justified by evidence, keep
collateral damage small, and use no gradient training. We measure how often it succeeds, how big the effect is, what
the edit costs elsewhere, and how its edits compare with known editing methods.

## 2. Targets

### Recommended

| Tier | Model | Why |
|---|---|---|
| 1 (main) | **GPT-2 small** `openai-community/gpt2` @ `607a30d7`, MIT, 124M | Fast: a full dev eval suite takes about 10 s on MPS [ran]. It is the best-documented model in interpretability, so several behaviours have **known ground-truth mechanisms** (induction heads, ROME-style fact storage in mid-layer MLPs). It has a **real failure mode**: greedy decoding loops on 16/16 dev prompts [ran], a close analogue of the paper's thinking loops. |
| 2 (paper target) | **Qwen3.5-0.8B** @ `2fc06364`, Apache-2.0, ungated | The paper's own target, with a documented loop pathology (its model card says so) [primary]. It loads in the harness and the paper's edit is imported hash-exactly [ran]. Generation is about 195 tok/s at batch 16 on MPS [ran], so experiments must lean on teacher-forced probes and **capped** generation checks. |

### Considered and rejected

- **Gemma 3 270M IT.** This was the paper's negative control and its sycophancy target. It is **gated** on Hugging
  Face (manual licence acceptance plus a token) [ran: `gated=manual`]. Accepting the licence and putting a token on this
  machine is your decision, not mine. Also, the paper found no editable obstacle in it for math.
- **Qwen3-0.6B** (Apache, ungated, standard attention, so simpler internals than Qwen3.5's DeltaNet layers). A
  reasonable stand-in for tier 2. Rejected for now because the paper's edits and logs are for Qwen3.5, which gives us
  a positive control, and because I have not checked whether Qwen3-0.6B loops.
- **Pythia / TinyStories models.** Good interpretability targets, but they add nothing over GPT-2 for a first round
  [inference].
- **Toy networks from the paper (Exp 1-5).** These are released as `.npz` and are useful as a "read weights" warm-up,
  but they are not language models. They are optional (§5, E0b).
- **Anything ≥ 1B on this Mac.** Generation-heavy evaluation is too slow with HF on MPS (see the Qwen numbers). MLX
  would be faster, but I have not verified it here (open question).

## 3. Tasks and metrics (all implemented in `src/wsp/evals.py`; dev and test splits)

| Task | What it measures | Main metric(s) | Used as |
|---|---|---|---|
| `ppl` | WikiText-2 test, 16k tokens per split, 512-token windows | perplexity | collateral damage (all experiments) |
| `induction` | 64 random-token sequences repeated once | second-half loss and top-1 | E1 target |
| `facts` | 31 short factual completions | top-1 accuracy, mean target log-prob | collateral damage |
| `counterfact` | the edit "Eiffel Tower → Rome" | rewrite / paraphrase success (P(new) > P(true)); neighbourhood success (P(true) > P(new) on other subjects) | E2 target |
| `repetition` | greedy 128-token continuations of 16 neutral prompts | rep-4 (share of repeated 4-grams), loop share (rep-4 > 0.5) | E3 target |
| `scripts/qwen_loop_check.py` | GSM8K-train problems, thinking on, paper sampling minus presence penalty, capped tokens | accuracy, share cut at the cap, share that closed `</think>`, tail rep-4 | E4 target |

Report for every submitted edit: `wsp diff` (entries changed, % of the model, delta rank), dev and test metrics for
the target task, and dev and test `ppl` and `facts`.

**Noise.** Every GPT-2 task is deterministic (greedy decoding, fixed seeds); rollback reproduces the metrics
bit-for-bit [ran]. Differences are therefore not sampling noise, but small sets (16 prompts, 15 facts) still have
coarse granularity. Treat a one-item change as nothing. For Qwen, sampling noise is about 50/√n points (the paper's
rule of thumb), so use paired original-vs-edited comparisons on the same problems.

## 4. Protocol for the editing agent (adapted from the paper's Exp 8 prompt)

1. **Budget.** Forward passes are counted by the harness: each `wsp run` and each `wsp eval --split dev` costs 1. Each
   experiment card fixes a budget (`wsp init --budget N`). Static weight analysis (`info`, `stats`, `show`, `svd`,
   `export` and numpy on exported tensors) is free.
2. **No training.** No backward passes, autograd, optimizers, or iterative fitting of weights to a loss. Do not run the
   model outside `wsp run` / `wsp eval` (no simulating the network in numpy on inputs).
3. **Protected tensors.** Embeddings, positional embeddings and the final norm (`--protect` at init). No edit that only
   scales a whole tensor by a constant. "Global knobs" (sharpening or flattening the output distribution) are out of
   scope.
4. **Diagnose, then localise causally.** A component being active is not enough; show that changing it changes the
   behaviour, at several strengths (dose-response). Include a **norm-matched random control** edit before claiming a
   mechanism.
5. **Dev for choosing, test for the experimenter.** The agent uses `--split dev` only. The experimenter runs
   `--split test` after the agent submits.
6. **Keep notes.** Checkpoint good states (`wsp checkpoint`). At the end, leave the run at the submitted edit and
   write `runs/<run>/final.md` covering what changed, why, the evidence, and the measured dev effect.
7. **Practical.** Print summaries, not whole arrays. Do not dump raw vocabulary lists (the paper had a turn refused by a
   content filter because of this [primary]).

Enforcement today is the budget counter, the protect list and the shape/finite checks. Claude Code has a shell and
could break rules 2 and 5. Whether to enforce them technically is an open question (§7).

## 5. Experiment ladder (in order of difficulty)

Every experiment starts from a fresh `wsp init`, records `wsp eval --model original` (dev) as its own baseline, and
finishes with the experimenter running `--split test` on both `original` and `edited`.

**E0: harness familiarisation (budget 5).** Load, inspect, probe "The Eiffel Tower is located in the city of", make
any edit, evaluate, roll back, and confirm the baseline numbers return. Success: correct use of the tools; bit-identical
metrics after rollback.
*E0b (optional): the paper's Exp 1 network* (`exp1_which_game/networks/*.npz`): name the game from the weights alone.

**E1: localise and switch off in-context copying (induction) in GPT-2 (budget 15).** The ground truth is known from
the literature: induction heads around layers 5-7 (e.g. 5.1, 5.5, 6.9, 7.10) [inference/literature; not re-verified
here]. Task: raise `induction.loss_second_half` as far as possible while keeping the `ppl` increase small. Report the
heads found and compare them with an attention-pattern ground truth computed by the experimenter.
Reference point from the smoke test [ran]: zeroing heads 5.1 and 5.5 moved second-half loss 0.218 → 0.758 and top-1
0.970 → 0.931, while perplexity went 37.8 → 79.0. That is a crude edit with a big collateral cost, which is precisely
what E1 asks the agent to do better than.

**E2: rewrite one fact (Eiffel Tower → Rome) in GPT-2 (budget 10).** Success: rewrite and paraphrase success ≥ 0.67 on
test, neighbourhood success no lower than baseline (dev 1.0, test 0.75), `ppl` within +2%. **Baseline method: ROME**
(rank-one update to a mid-layer MLP `c_proj`), with the same metrics. The harness's `rank1` operation is the right
primitive, but ROME itself is not yet implemented (open question).

**E3: stop GPT-2's greedy loops (budget 25), the analogue of the paper's main result.** Task: lower `repetition.rep4`
and `loop_share` (dev baseline 0.79 / 1.00; test 0.66 / 0.875) with `ppl` within +5% and `facts` top-1 not lower.
Baselines: decode-time `repetition_penalty` and `no_repeat_ngram_size` (no weight change; they show how far a decoding
fix gets); optionally a short unlikelihood fine-tune as a gradient baseline. *Watch for* edits that cut repetition by
degrading the model (perplexity up) or by making it emit EOS early (`mean_len` down).

**E4: Qwen3.5-0.8B thinking loops, small scale (budget 25).** Rediscover a loop-related direction using teacher-forced
probes (repetitive vs productive continuations) and capped generation passes. Positive control: the paper's p50c edit
(`scripts/import_paper_edit.py`). Negative control: a norm-matched random direction in the same columns. Metric:
`scripts/qwen_loop_check.py` (cut share, closed-think share, accuracy) on GSM8K-train problems not used for choosing
the edit. Slow: each check of 32 problems × 3,072 tokens × 2 models takes minutes (see §6).

**Edit budget beyond passes.** No hard cap on the number of weights changed, but `wsp diff` must be reported. The
paper's edits ranged from 0.016% of weights (p50c) to dense rank-1 changes to 30 matrices (p10c) [primary].

## 6. What was run to verify the scaffold (exact commands, real output)

The environment was built with `uv sync` (Python 3.12.13, torch 2.14.1, transformers 5.21.0, MPS available). Then
`uv run --group dev pytest -q` gave `5 passed`.

**GPT-2 smoke test** (`runs/gpt2-smoke`, created with `--budget 20 --protect 'wte|wpe|ln_f'`):

```
uv run wsp info runs/gpt2-smoke          -> GPT2LMHeadModel  params: 124,439,808  device: mps
uv run wsp run runs/gpt2-smoke --prompt "The Eiffel Tower is located in the city of" --target " Paris" --hidden --capture transformer.h.5.mlp
                                          -> top-1 ' Paris' p=0.0638; saved hidden_states [13,2,12,768], h.5.mlp [2,12,768]
uv run wsp eval runs/gpt2-smoke --split dev   (untouched model: the baseline)
uv run wsp eval runs/gpt2-smoke --split test  (untouched model: the held-out baseline)
```

**Baseline, untouched GPT-2 small** (deterministic; the test row for `counterfact` is after fixing the neighbourhood
set, see §8):

| Metric | dev | test |
|---|---|---|
| ppl (WikiText-2, 16,352 tokens) | **37.779** | **33.032** |
| induction: second-half loss / top-1 | 0.2176 / 0.9704 | 0.2416 / 0.9674 |
| facts top-1 (n) | 0.600 (15) | 0.5625 (16) |
| counterfact rewrite / paraphrase / neighbourhood success | 0.0 / 0.0 / 1.0 | 0.0 / 0.0 / 0.75 |
| repetition rep-4 / loop share | 0.7905 / 1.0 | 0.662 / 0.875 |

**Trivial edit, then rollback:**

```
uv run wsp checkpoint runs/gpt2-smoke before-ablation
uv run wsp edit runs/gpt2-smoke zero transformer.h.5.attn.c_proj.weight --index "[64:128, :]"     # head 5.1
uv run wsp edit runs/gpt2-smoke zero transformer.h.5.attn.c_proj.weight --index "[320:384, :]"    # head 5.5
uv run wsp diff runs/gpt2-smoke   -> 98,304 of 124,439,808 weights (0.0790%), delta rank 128
uv run wsp eval runs/gpt2-smoke --task induction --task ppl --task facts
                                  -> ppl 79.004, induction 2nd-half loss 0.7577, top-1 0.9310, facts 0.667
uv run wsp edit runs/gpt2-smoke zero transformer.wte.weight --index "[0:1, :]"
                                  -> ProtectedTensor: refused
uv run wsp rollback runs/gpt2-smoke before-ablation
uv run wsp eval runs/gpt2-smoke   -> every metric identical to the baseline to the last digit
uv run wsp rollback runs/gpt2-smoke -1   -> the ablation is back (undo of the rollback works)
uv run wsp reset runs/gpt2-smoke
```

An independent check without the harness (plain transformers on CPU, the same text windows) gave ppl 37.778 for the
base model and 79.002 after the same ablation. That matches the harness on MPS (37.779 / 79.004).

Budget enforcement: a run with `--budget 1` refused the second `wsp run` with `BudgetExceeded: forward-pass budget
exhausted: 1/1 used`.

**Qwen3.5-0.8B** (`runs/qwen35-smoke`, `runs/qwen35-paper-p50c`):

```
uv run wsp info runs/qwen35-smoke   -> Qwen3_5ForCausalLM  params: 752,393,024  device: mps
generation throughput (thinking on, paper sampling), HF on MPS: batch 1 = 42 tok/s, batch 16 = 195 tok/s
uv run python scripts/import_paper_edit.py runs/qwen35-paper-p50c external/lm_understand-logs/exp8_math/runs/p50c/edit
  -> model.layers.15.mlp.down_proj.weight: hash OK, 65,260 entries changed
     model.layers.19.mlp.down_proj.weight: hash OK, 64,928 entries changed
uv run wsp diff runs/qwen35-paper-p50c -> 130,188 of 752,393,024 (0.0173%), 64 columns per tensor, delta rank 1 each
```

**Small behavioural check of the paper's p50c edit** [ran]:
`uv run python scripts/qwen_loop_check.py runs/qwen35-paper-p50c --n 32 --max-tokens 3072`. This uses GSM8K-train
problems 0-31, thinking on, temperature 0.6 / top-p 0.95 / top-k 20, no presence penalty, and a 3,072-token cap.

| | accuracy | cut at cap | closed `</think>` | tail loop share (rep-4 of last 1,024 tokens > 0.5) | mean length | time |
|---|---|---|---|---|---|---|
| original | 1/32 | 31/32 | 1/32 | **22/32 (0.69)** | 3,008 | 543 s |
| p50c edit | 1/32 | 31/32 | 1/32 | **7/32 (0.22)** | 3,045 | 592 s |

How to read it [inference]: the edit visibly reduces *looping* in the generated text, which is consistent with the
paper's mechanism. But at a 3,072-token cap almost every answer is still cut off, because this model thinks for
thousands of tokens (the paper's edited model had a median of 9,172 tokens on correct answers [primary]). So this
check **cannot show the accuracy gain**. Reproducing accuracy effects needs caps of about 16-32k tokens, which means
hours per pass at the measured ~180 tok/s. For E4, the loop share at a short cap is the practical metric, with
accuracy reserved for a few long, expensive confirmation runs. The sample is n=32 in each arm, unpaired sampling
(batched generation does not share random numbers per problem), and a single seed.

## 7. Open questions for you to decide

1. **Enforcement of the rules.** Claude Code has a shell, so it could write a training loop or peek at `--split test`.
   Options: (a) trust plus audit (the journal and transcript); (b) a Claude Code permission allowlist restricting Bash
   to `uv run wsp *` and `uv run python scripts/*`; (c) a hook that blocks `backward(`, `optim`, `--split test`. I have
   not changed any Claude Code settings.
2. **Editor configuration.** The model id and reasoning effort for "Claude Fable 5.2", and whether to give it the full
   protocol as a system prompt or a `CLAUDE.md`. Context-length handoffs (the paper restarted every 40 turns) will
   matter for long runs.
3. **ROME baseline.** Implement a minimal ROME for GPT-2 ourselves (about 150 lines, my recommendation), vendor
   EasyEdit / the original `rome` repo (heavy, untrusted code to review), or skip it.
4. **Gradient baseline.** Do we want a fine-tuning comparison (e.g. a few hundred steps of unlikelihood training for
   E3)? It is a useful reference for "how much would gradients get you", but it is extra scope.
5. **Gemma 3 270M.** It is gated. Do you want to accept its licence and provide an HF token to replicate the paper's
   negative result?
6. **MLX for the Qwen tier.** Generation-heavy work on Qwen3.5 would be several times faster with MLX (likely, not
   verified). Adding `mlx-lm` means a second inference path whose numerics differ from torch.
7. **How many repeats.** The paper's results varied a lot by prompt version and budget. Decide up front how many
   independent attempts per experiment (I suggest 3) and report all of them, including runs that submit nothing.
8. **Disk.** The data volume is 92% full (154 GiB free). Each extra model costs 0.5-2 GB, plus activation dumps under
   `runs/`.

## 8. Known limitations of the scaffold

- Each CLI call reloads the model (about 1-2 s for GPT-2, a few seconds for Qwen). Fine for step-by-step use. A
  persistent kernel would be faster for long sessions.
- History snapshots store the whole overlay at every step, so disk use grows with (number of edits × size of edited
  tensors). This is trivial for GPT-2 but worth watching if many large Qwen tensors are edited.
- `edit set` / `edit add` can technically express a whole-tensor rescale; only `edit scale` refuses one.
- Task sets are small and hand-written (facts 31, counterfact 1 edit with 6 paraphrases and 8 neighbours, 32
  repetition prompts). They are good enough to show effects, too small for fine claims. I fixed one defect found
  while verifying: a neighbourhood item whose true answer equalled the edit target.
- `qwen_loop_check.py` omits the paper's presence penalty (HF `generate` has no direct equivalent) and caps tokens far
  below 32,768, so its numbers are **not comparable** to the paper's.
- `--capture NAME` saves a module's output, and `--capture NAME:in` its input. For example
  `transformer.h.5.mlp.c_proj:in` gives GPT-2's 3,072 MLP neuron activations [ran], and
  `model.layers.15.mlp.down_proj:in` is the Qwen equivalent. I listed the Qwen capture points (`wsp modules`) but did
  not run a capture on Qwen.
