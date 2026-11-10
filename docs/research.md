# Research notes: LLMs editing neural-network weights by hand

Prepared 2026-11-10 as groundwork for an experiment in which Claude (driven through Claude Code) edits a small
model's weights directly, with tools and without gradient training.

**How to read the evidence labels**

- **[primary]**: read in a primary source (the authors' blog post, their logs repository, or the Hugging Face model
  cards), linked inline.
- **[ran]**: checked by running something on this machine. Commands and outputs are in `docs/experiment-design.md`
  §6 and in `runs/*/journal.jsonl`.
- **[secondary]**: reported only by press or blog coverage. Not confirmed.
- **[inference]**: my own reasoning or judgement.

## 0. Sources and what could not be reached

| Source | Status |
|---|---|
| Medium article, "MIT let an AI edit another AI's weights by hand" ([link](https://medium.com/@yash140498/mit-let-an-ai-edit-another-ais-weights-by-hand-5cc73feae199)) | **Not reached.** Medium returned HTTP 403. The freedium mirror failed with a TLS certificate error, scribe.rip returned 404, and web.archive.org is blocked for the fetch tool. A title search found only other coverage of the same work. **So I cannot say where this particular article differs from the paper.** §6 instead compares the primary source with the secondary coverage I could read. |
| Blog post (the "paper"): Takehi, Teoh, Zhang, Khattab (MIT CSAIL), *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, 2 Nov 2026, <https://rikiyat.github.io/blog/weight-space-programming/> | Read [primary]. **It is a blog post, not an arXiv paper.** I found no arXiv version. It is not peer reviewed. |
| Logs repository: <https://github.com/RikiyaT/lm_understand-logs> (commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`) | Sparse-cloned for reading into `external/lm_understand-logs/` (gitignored, treated as untrusted, none of its code executed). I read the top-level README and LICENSE, `harness/`, all of `exp8_math/` that matters (README, task prompt, notices, final answers, edit manifests, `code/apply_edit.py`, results excerpt), `exp8_math/gemma_followup/README.md` and `exp7_sycophancy/README.md` [primary]. |
| Edited checkpoint: <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming> | Model card read [primary]. |
| Demo Space: <https://huggingface.co/spaces/akhaliq/weight-space-programming-demo> | Linked from the post. Not opened. |
| Base model card: <https://huggingface.co/Qwen/Qwen3.5-0.8B> | Read [primary]. |
| Works the post cites: [arXiv 2102.11174](https://arxiv.org/abs/2102.11174) (Schlag et al., linear transformers as fast-weight programmers), [arXiv 2301.05062](https://arxiv.org/abs/2301.05062) (Tracr), Karpathy, ["Software 2.0"](https://karpathy.medium.com/software-2-0-a64152b37c35) | arXiv abstracts read. Background only. |
| Critiques and coverage: [buildmvpfast](https://www.buildmvpfast.com/blog/astra-weight-editing-qwen-misleading-rebuttal-2026) (403, not read), [korben.info](https://korben.info/en/astra-tweaked-qwen-weights-finish-reasoning.html), [letsdatascience](https://letsdatascience.com/blog/gpt-6-astra-edited-0-016-percent-qwen-weights-math-sixfold), [VentureBeat](https://venturebeat.com/technology/like-gene-editing-for-neural-networks-mit-researchers-used-gpt-6-astra-to-edit-qwens-open-weights-directly-boosting-math-performance), [The Register](https://www.theregister.com/a/5307105) | Seen only through search-result summaries [secondary]. |

## 1. What the work actually did

**Definition** [primary]: "Weight-Space Programming" means a general-purpose LLM reads and writes a network's weights
directly, "not through an optimizer, and in particular not with gradient descent", guided by a natural-language spec.

**Editor** [primary]: "Astra", which the logs README identifies as OpenAI's `gpt-6-astra`. It was run through the
Codex CLI (`codex exec`), with all built-in tools switched off (shell, file editing, web, sub-agents). Reasoning effort
was `high` for Exp 1, 2 and 4, `xhigh` for Exp 3, 5, 6 and 7, and `max` for Exp 8. Where code was allowed, the model
wrote **one Python block per turn**. The authors' own sandbox ran it (bubblewrap, no network, a persistent working
folder, a time limit), and the printed output came back as the next message. All 1,125 turns were audited as
tool-free.

**Eight experiments** [primary] (Exp 1-7 summarised; Exp 8 is the one that matters for us):

| # | Target | Size | Task | Interface | Result |
|---|---|---|---|---|---|
| 1 | PPO policy MLP 500→32→32→5 (tanh) | 17,253 weights | name the game (Breakout, Pong, Pac-Man, Freeway) | weights pasted as text, no tools | 4/4 correct |
| 2 | DDPM MLP 160→64→64→144 | 23,824 | name the shape the model generates | weights as text, no tools, no forward passes | 3/3, but drew outlines although the models generate filled shapes |
| 3 | char-level transformer, 4 layers × 6 heads, d=384 | 7.16M | name the training language from the middle blocks only; find copying heads | Python (≤100 runs), no forward passes | language 6/6; copying heads exactly right |
| 4 | Kinetix PPO actor (car on ramp) | 171,664 | fix a policy that never succeeds | weights, a scene description and one recorded episode; a one-shot reply | added a side path (3.5% of weights); 1,024/1,024 episodes succeed |
| 5 | MinAtar MLP 600→32→32→6, from zeros | small | write a policy from scratch | rules and source code, no data, one reply | Breakout 517.4 vs PPO-500M 25.3; Seaquest 53.1 vs 88.9; Asterix 53.2 vs 55.2 |
| 6 | 9×9 Go MLP 243→512×6→82, from zeros | 1.48M | write a Go player | Python sandbox with 0-50 budgeted forward passes | 0 passes: beats small PPO nets; 50 passes: beats an equal-size PPO net trained on 20M games |
| 7 | Gemma 3 270M fine-tune pairs (honest vs sycophant) | 270M | which model is sycophantic? | weights alone, then an attention-only probe tool (≤16 texts, ≤20 runs) | **weights alone: failed 4/4**; with the attention tool: 3/3 (head 10.2) |
| 8 | **Qwen3.5-0.8B, thinking on** | 0.8B | **raise math accuracy by editing weights** | Python kernel plus a forward-pass tool with a 10/25/50-pass budget | MATH-500 7.9 → 48.1 (50 passes) |
| 8b | Gemma 3 270M IT | 270M | the same task, on a model that rarely loops | the same harness | **no real gain** (6.8 → 7.0-7.7) |

### 1.1 Experiment 8 in detail (all [primary], from `exp8_math/README.md` and `runs/p50c/turns/t001/prompt.txt`)

**Target.** The text part of `Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`): 24 layers,
width 1,024. The layers alternate Gated DeltaNet linear attention and gated full attention (layers 3, 7, 11, 15, 19
and 23). The MLP is SwiGLU with width 3,584. The vocabulary has 248,320 tokens, and the embedding is tied to the output
layer. The model card itself warns that this model "is more prone to entering thinking loops" in thinking mode.

**What the editor had.**
1. A persistent Python kernel: CPU only, no internet, at most 300 code runs, 900 s and 96 GB per run, output cut at
   10,000 characters. It had these preloaded functions: `NAMES`, `load(name)` (the current edit or the original),
   `load_original`, `save_edit(name, tensor)`, `drop_edit`, `edited()` and `tok`. Gradients were disabled.
2. Practice data: 7,469 GSM8K-train and 7,275 MATH-train problems with solutions, plus 714 AIME problems with answers
   only. No test data.
3. A **forward-pass tool** with a budget of **10, 25 or 50 passes**. A pass is one of two kinds:
   - a *generation pass*: up to 512 answers with the evaluation's sampling, run on the original model, the edited
     model, or `both` (paired comparison on the same problems with fixed seeds, reporting the difference and its SE).
     It can optionally save residual-stream and MLP activations, averaged over each answer.
   - a *teacher-forced pass*: up to 1,024 texts and 524k tokens, returning log-probs, top-k, hidden states, MLP
     activations and attention for full-attention layers.

**Rules.** No backward passes, autograd, optimizers or iterative fitting. No running the model in the kernel; only the
forward-pass tool runs it. No internet. `save_edit` refused edits to the final norm and the embedding/LM head, and any
edit that just multiplies a whole tensor by a constant. The prompt explicitly asks for causal localisation
(dose-response, not just correlation), paired comparisons, enough statistical power, and held-out confirmation.
Experimenters' notices sent during the runs corrected tool facts and changed the submission rule to "submit the best
edit found, even if not significant".

**Evaluation** (never shown to the editor). vLLM; thinking on; temperature 0.6, top-p 0.95, top-k 20,
**presence penalty 1.5**; up to 32,768 new tokens; grading of the last `\boxed{}` with math-verify. Samples per
problem: MATH-500 4, GSM8K 2, AMC23 16, Minerva/OlympiadBench/GPQA 4, IFEval 2, MMLU-Pro 1 (a fixed 1k subset).

**Edits and results.**

| Run | Passes | Turns / hours | Final edit | MATH-500 | GSM8K | IFEval | Cut at token limit |
|---|---|---|---|---|---|---|---|
| unedited | - | - | - | 7.9 | 13.8 | 21.0 | 93.0% |
| p10c | 10 | 74 / 11.6 h | remove a "repetition" direction from 30 output matrices (attention and MLP, layers 8-22), W' = W − 1.5·u(uᵀW) | 12.3 | 25.7 | 49.0 | 87.2% |
| p25c | 25 | 86 / 8.5 h | zero attention head 7 of layer 15 (its 256 output columns) | 14.1 | 25.1 | 39.9 | 86.4% |
| p50c | 50 | 158 / 20.3 h | subtract the loop direction from the 64 most-writing output columns of the MLP `down_proj` in layers 15 and 19 (131,072 weights) | **48.1** | **67.1** | **62.3** | 33.1% |

Other benchmarks for p50c: AMC23 0.5 → 23.4, Minerva 1.5 → 25.4, OlympiadBench 1.3 → 21.8, MMLU-Pro 40.3 → 45.5,
GPQA-Diamond 22.7 → 36.6. The post's figure shows 24.5 for the unedited GPQA score, which was an interim value; the
README corrects it to 22.7.

**What the edit is, mechanically.** The gain comes from **removing a failure mode (endless thinking loops)**, not from
adding mathematical skill. The authors say so themselves: IFEval rose too, and on Gemma 3 270M, which rarely loops
(8.5% of answers cut vs 93%), no edit moved MATH-500 by more than about a point. Their conclusion is that "editing
weights can remove obstacles but cannot add missing abilities". [primary]

**Failure modes and caveats reported by the authors** [primary]:
- Exp 7: could not tell the sycophantic model from the honest one by weights alone (0/4).
- Exp 2: drew outlines rather than filled shapes.
- Exp 8: at 10 and 25 passes looping barely fell. Results were **not monotonic in budget** and depended heavily on the
  prompt version. `eval/RESULTS_excerpt.md` shows unreported runs: a 5-pass v3 run and a 10-pass v4 run **submitted no
  edit**, and v5 rows are empty. That is a selection effect worth keeping in mind. On Gemma the gains were within
  noise.
- Long wall-clock time (8-20 h per run), context-length limits ("handoffs" to a fresh conversation every 40 turns),
  and one turn refused by the provider's content filter because vocabulary strings were printed.

## 2. What is released and what is not

| Item | Released? | Licence |
|---|---|---|
| Full logs of every reported run (prompts, replies, code and outputs, forward-pass requests and summaries, notebooks) | yes [primary] | CC BY 4.0 |
| Final networks and edits. Exp 8 edits as **exact bf16 diffs** against the public base, plus `apply_edit.py` that checks sha256 hashes | yes [primary]; **hash-verified with our own importer** [ran] | CC BY 4.0 (data), MIT (scripts) |
| Evaluation code for Exp 8 (`evaluate.py`, `mathqa.py`, `profiles.py`, `merge.py`) | yes, but it needs vLLM (CUDA); the IFEval and multiple-choice graders it imports are **not** included [primary] | MIT |
| Practice problem files (GSM8K/MATH train, AIME) | yes [primary] | each dataset's own licence |
| Go-experiment sandbox (`harness/sandbox.py`, bubblewrap, so Linux only) and the Codex config | yes [primary] | MIT |
| **The Exp 8 agent harness** (kernel bridge, forward-pass server, prompt driver) | **no**: "the agent harness code (the prompts, sandbox and forward-pass server...)" is listed under *Not included* [primary] | - |
| Exp 7 Gemma fine-tunes (17 GB) | no; the training script and data are there to rebuild them [primary] | - |
| Edited Qwen checkpoint on HF (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`) | yes; rebuilt from the logs by Ahsen Khaliq [primary] | Apache-2.0 |
| A benchmark or harness for weight-space programming | **announced as future work**, not released [primary] | - |

**Build on it, vendor it, or write our own?** [inference] **Write our own small harness**, and borrow three things from
the release:
1. **There is nothing runnable to build on for an LLM-editing loop.** The Exp 8 harness is unreleased. The released
   sandbox is a Linux bubblewrap wrapper for the Go experiment. The evaluation stack is vLLM/CUDA, which does not run on
   this Mac.
2. **Our editor's interface is different anyway.** Astra had no tools of its own and acted one code block per turn
   through the authors' harness. Claude Code has a shell, so the natural tool surface is a CLI plus a Python library
   that it calls step by step. The paper's *rules* (budgeted forward passes, no gradients, protected tensors, paired
   comparisons, a DEV/HOLDOUT split) carry over as protocol. The *plumbing* does not.
3. **Borrow these, with attribution:** (a) the Exp 8 task prompt as the model for our protocol text; (b) the practice
   files (GSM8K/MATH train) for the Qwen tier; (c) the released edits as **positive controls**. Our
   `scripts/import_paper_edit.py` re-implements their diff format, about 40 lines, and reproduces the p50c tensors
   bit-exactly. Vendoring their MIT scripts would be allowed, but we do not need them.

## 3. What a small local reproduction needs

**Hardware and tooling on this machine** [ran]: Apple **M5 Max**, **128 GB** unified memory, 18 CPU cores, macOS
26.6.1 (Darwin 25.6). uv 0.12.3; Python 3.12.13 (uv-managed) in the project venv; torch 2.14.1 with MPS available;
transformers 5.21.0. Disk has 154 GiB free, and the volume is 92% full, so keep the model count small. No `HF_TOKEN`
is set.

**Target models** (details and rejected alternatives in `docs/experiment-design.md`):
- **GPT-2 small** (`openai-community/gpt2`, MIT, 124M, revision `607a30d7`). It is very well studied (induction heads,
  IOI, ROME-style fact storage), quick (a full dev evaluation suite takes about 10 s) [ran], and it has a **visible
  failure mode, greedy-decoding loops**, that mirrors the paper's thinking loops (rep-4 0.79, loops on 16/16 dev
  prompts) [ran].
- **Qwen3.5-0.8B** (Apache-2.0, ungated, the paper's own target). It loads in the harness as `Qwen3_5ForCausalLM`,
  752M text parameters [ran]. Generation runs at about **195 tok/s at batch 16 and about 42 tok/s at batch 1** on MPS
  with HF transformers [ran]. One paper-sized generation pass (512 × up to 32k tokens) would take **roughly a day**, so
  a faithful replication of Exp 8 is not practical here [inference from the measured speed]. Inspection,
  teacher-forced probing and small capped generation checks are practical.

**Datasets.** WikiText-2 (test split, pinned revision) for collateral perplexity; synthetic repeated random tokens for
induction; small hand-written fact, counterfact and prompt sets in `data/tasks/`; GSM8K/MATH-train practice files from
the logs repo for Qwen.

**Metrics.** Perplexity (collateral damage); induction loss and top-1; fact top-1 and log-prob; CounterFact-style
efficacy, paraphrase generalisation and neighbourhood specificity; rep-4 and loop share under greedy decoding; for
Qwen, accuracy, the share of answers cut at the token cap and the share that closed `</think>`, at a reduced cap. Also
report the **number of weights changed and the rank of the delta** (`wsp diff`), as the paper does.

**Baselines.** The untouched model; a **norm-matched random edit** of the same shape (a control the paper also used
inside p50c, per its final answer); for the fact edit, **ROME** (rank-one MLP update; not yet implemented, see open
questions); for loops, **decode-time repetition penalties** (no weight change) and optionally a short fine-tune as a
gradient baseline; for Qwen, the **paper's own released edits** (positive control).

**Tool surface** (built as `wsp`, see the README): load and list tensors; stats and row/column norms; slices; SVD;
export to `.npy`; forward passes with top-k, target log-probs, generation, and capture of any module's output, the
residual stream and attention; edits (set, add, scale or zero a slice, rank-1, project out a direction); diff against
the original; evaluation on dev/test splits; a forward-pass budget; append-only history with checkpoints, rollback and
reset; and export of the edited model.

## 4. Related work, only where it changes the design

- **ROME / MEMIT** (Meng et al., 2022: [arXiv 2202.05262](https://arxiv.org/abs/2202.05262),
  [arXiv 2210.07229](https://arxiv.org/abs/2210.07229)). Facts can be rewritten by a closed-form rank-one update to a
  mid-layer MLP output matrix. *Design impact:* the fact-rewrite experiment has a strong non-LLM baseline and standard
  metrics (efficacy, paraphrase, neighbourhood), and the harness's `rank1` operation is exactly the primitive ROME uses.
  [cited from memory; abstracts not re-read this session]
- **Refusal / feature direction ablation** (Arditi et al., 2024, [arXiv 2406.11717](https://arxiv.org/abs/2406.11717))
  and activation steering. A behaviour mediated by one residual direction can be removed by orthogonalising the
  writing matrices against it. That is exactly the shape of the paper's p10c and p50c edits (W' = W − s·u uᵀW).
  *Design impact:* `project-out` is a first-class operation, and a "difference of mean activations" probe should be
  easy (capture the residual stream, then numpy). [cited from memory]
- **Induction heads / transformer circuits** (Elhage et al., 2021; Olsson et al., 2022). Astra used Elhage-style
  analysis unprompted in Exp 3 [primary]. *Design impact:* GPT-2 small's induction circuit is a target with known
  ground truth for the first localisation exercise. [cited from memory]
- **Tracr** ([arXiv 2301.05062](https://arxiv.org/abs/2301.05062), cited by the post). Compiled transformers with
  known algorithms. A possible later source of ground-truth targets; not needed now.

## 5. Things we verified by running (summary; full output in `docs/experiment-design.md` §6)

- The environment installs with `uv sync`; there are 5 offline unit tests (`uv run --group dev pytest`).
- GPT-2 small: load, inspect, probe with capture, edit, evaluate, roll back. **Rollback reproduces the baseline
  metrics bit-for-bit.** An independent plain-transformers CPU script reproduces the harness's baseline (37.778 vs
  37.779) and its ablation perplexity (79.002 vs 79.004).
- Qwen3.5-0.8B: loads and lists tensors; generation throughput measured; the paper's p50c edit imported with **both
  sha256 hashes matching**, 65,260 + 64,928 entries changed (manifest: the same), delta rank 1 per tensor. On 32
  GSM8K-train problems with a 3,072-token cap, the edit cut looping answers from 22/32 to 7/32, but accuracy (1/32 in
  both arms) and cut-off share (31/32 in both) did not move, because the cap is far below how long this model thinks.

## 6. The primary source compared with the coverage

The Medium article itself could not be read (see §0). Comparing the post and logs with the coverage I *could* see:

- Coverage calls it a "paper". It is a **blog post plus a logs repo**, not peer reviewed, with no arXiv version found.
- "Changed 0.016% of the weights" matches: 131,072 weights against the 0.8B *including the vision tower*. Against the
  752M text parameters it is 0.017% (130,188 entries actually differ after bf16 rounding) [ran].
- "Math went up 6×" is true for MATH-500 under the paper's protocol, but the mechanism is **loop removal**. 93% of the
  base model's answers ran to the 32k token limit. The base model card already warns about loops, and the authors
  themselves frame it as removing an obstacle. Critics (buildmvpfast, Honnibal per secondary reports) push the same
  point harder [secondary].
- The "independent re-run at 51.0%" on the HF card was done by the team that built the checkpoint (Ahsen Khaliq,
  credited by the authors), on a 25-problem smoke test for the base model. It is not an independent replication
  [primary: model card].
- Most coverage leaves out the **non-monotonic budget results, the unreported runs that submitted no edit, the
  sensitivity to prompt version, and the presence penalty of 1.5 in the evaluation protocol** [primary: RESULTS_excerpt].
- One secondary claim (korben.info) says the edit costs about 2 points with thinking off. Another mentions a "Willison
  replication with Qwen 3.8 27B". **Neither is confirmed**, and the second does not match the model in the post
  [secondary].
- "GPT-6 Astra" is confirmed by the logs README as OpenAI's `gpt-6-astra` [primary]. I know nothing else about that
  model.
