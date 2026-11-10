# Stall prediction stand-in: can a detector see a stall coming?

**Result: only weakly.** From the start of a run, every predictor tried is between chance and an AUC of about 0.70,
and the model-based ones are not clearly better than simple transcript statistics.

## Setup
- **Runs:** 256 runs of unedited Qwen3.5-0.8B thinking through GSM8K practice problems (problems 64-319), sampled at
  temperature 0.6, capped at 2,048 generated tokens (`scripts/stall_runs.py`). All 256 hit the cap.
- **Stall:** the last 768 tokens are mostly repeated 4-grams (rep-4 > 0.5). 87 of 256 runs stall (34%). Among stalls
  where a loop start could be located, the median start is token 1,280 (earliest 128).
- **Task:** given the problem plus the first K generated tokens, predict the stall. Runs already looping inside the
  first K tokens are dropped, so a positive is a stall that has not started.
- **Scoring:** held-out AUC (0.5 = chance, 1.0 = perfect), mean and spread over 8 random 70/30 splits by problem
  (`scripts/stall_predict.py`).
- **Predictors:** closed-form detector (`wsp.detector`), a trained logistic-regression probe on the same activations,
  and logistic regression on five transcript statistics with no language model (repetition so far, vocabulary
  variety, prompt length, digit count).

## Results (held-out AUC, mean +/- spread over splits)
| First K tokens | Transcript statistics | Best detector, GPT-2 reads Qwen's runs | Best detector, Qwen reads its own runs |
|---|---|---|---|
| 0 (problem only) | 0.52 +/- 0.07 | 0.54 +/- 0.07 | 0.62 +/- 0.05 |
| 128 | 0.49 +/- 0.04 | 0.62 +/- 0.06 | not run |
| 256 | 0.59 +/- 0.07 | 0.70 +/- 0.04 | 0.67 +/- 0.06 |
| 512 | 0.64 +/- 0.07 | 0.69 +/- 0.05 | not run |

"Best" is the highest of the settings tried at that K (GPT-2: 4 layers x 2 pooling methods; Qwen: 2 layers x 2),
so these columns are optimistic: picking the maximum of several noisy numbers inflates it. Full tables are in
`gpt2_reader.json` and `qwen_reader.json`. The trained probe was within about 0.03 of the detector in most
settings and never clearly ahead.

## Reading
- There is some signal once part of the run is visible (0.6-0.7 against 0.5), and averaging activations over the
  whole prefix works better than reading only the last token.
- A different model reading the transcript (GPT-2) did about as well as the model that produced it (Qwen). That is
  mildly encouraging for using an open model to read a hosted model's runs, but both are weak here.
- The closed-form detector again matches the trained probe, so the weak result is not a failure of the hand-built
  method; the information is not strongly present in these activations, or there is too little data to find it.
- Likely reasons, none tested: (1) with sampling, whether a run loops may be decided by chance after the prefix, in
  which case no predictor can do well; (2) 256 runs with about 60 stalls per fitting split is small for 768-1,024
  dimensional activations; (3) the label is noisy (22 of 87 stalls have no locatable loop start, and 12 non-stall
  runs looped at some point and recovered).

## Limits
- Not comparable to a real agent run: these are single-model monologues on short maths problems, not multi-step
  tool-using sessions. A wedge in an agent may leave much clearer traces (repeated tool calls, errors).
- One seed for generation; the Qwen reader was run at fewer settings because each pass is slow (about 100 s).
- No test of how reproducible a stall is for the same problem, which would give the ceiling for any predictor.
