# E1: switch off in-context copying (induction) in GPT-2 small

Editor: Claude Fable 5.2 in Claude Code. Budget 15 forward passes, 11 used (2 of them wasted on a script bug).
Run left at checkpoint `submitted`.

## Submitted edit
Mean-ablate five attention heads: 5.1, 5.5, 6.9, 7.2, 7.10. For each head, the 64 rows of
`transformer.h.L.attn.c_proj.weight` that carry its output are zeroed, and the head's average output on natural text
is folded into `transformer.h.L.attn.c_proj.bias`. 248,064 weights changed (0.199% of the model). No gradients.

## Evidence
1. **Diagnosis (1 pass).** Attention patterns on 24 repeated random-word sequences. Share of attention going to the
   token after the previous occurrence: 5.5 0.92, 6.9 0.90, 5.1 0.88, 7.10 0.88, 7.2 0.78; next best 10.7 0.52.
   Supporting heads: previous-token head 4.11 (0.99), duplicate-token heads 0.5, 3.0.
2. **Why not zero the heads.** Outside copying these heads put 0.87-0.99 of their attention on position 0 and emit a
   near-constant vector (head 5.1: mean norm 2.28, spread 0.11). Zeroing removes that constant too, which is why the
   scaffold's zero-ablation of 5.1+5.5 doubled perplexity. Keeping the constant in the bias avoids it.
3. **Dose-response and controls (dev split).**

| Edit | induction 2nd-half loss | 2nd-half top-1 | ppl | facts top-1 |
|---|---|---|---|---|
| none | 0.218 | 0.970 | 37.78 | 0.60 |
| mean-ablate 5.5 | 0.225 | 0.968 | 38.19 | 0.60 |
| mean-ablate 5.5, 5.1, 6.9, 7.10 | 8.786 | 0.040 | 42.62 | 0.60 |
| **mean-ablate all five (submitted)** | **9.963** | **0.010** | **43.82** | 0.60 |
| control: mean-ablate 5.3, 5.8, 6.2, 6.6, 7.5 | 0.263 | 0.968 | 39.01 | 0.60 |
| pin all five to position 0 (query/key edit) | 10.198 | 0.006 | 44.23 | 0.60 |
| control: pin 5.3, 5.8, 6.2, 6.6, 7.5 | 0.334 | 0.968 | 39.60 | 0.53 |

   One head alone does nothing (the heads are redundant); four collapse copying. The control uses the same number of
   heads in the same layers and the same number of weights, not a norm-matched random matrix.
4. **Second mechanism, same result.** "Pin" leaves the head's output weights alone and instead replaces its query with
   a constant along the position-0 key direction, so it can never look at the earlier copy. It gives almost the same
   induction loss and perplexity as mean-ablation, so the +16% perplexity is most likely the price of losing copying on
   WikiText itself (which repeats names and terms), not damage from the edit. That reading is an inference.

## Not done
- Test split not run (reserved for the experimenter):
  `uv run wsp eval runs/e1-induction --split test --task induction --task ppl --task facts` with `--model original`
  and `--model edited`.
- The mean vectors come from 8 short paragraphs I wrote; a larger sample might shave the perplexity cost slightly.
- Scripts and probe inputs are in `analysis/`.
