# E3: stop GPT-2 small's greedy-decoding loops

Editor: Claude Fable 5.2 in Claude Code. Budget 25 forward passes, all 25 used (one by accident: a repeated eval).
Run left at checkpoint `submitted`.

## Submitted edit
A "loop detector -> correction" written into one MLP: a rank-1 update to `transformer.h.6.mlp.c_proj.weight`
(W + 3.0 * u v^T) plus a matching change to its bias. 2,360,060 entries changed (1.90% of the model; rank 1). No
gradients.
- **u (detector):** difference between the average layer-6 MLP neuron pattern at loop positions (a 4-gram that already
  occurred, in the model's own greedy output) and at ordinary positions, multiplied by the inverse second-moment matrix
  of ordinary-text neuron patterns (3,266 positions: 16 written passages and 32 passages sampled from the model; ridge
  1.0). Held-out response: loop 0.90 +/- 0.38, ordinary 0.01 +/- 0.09.
- **v (correction):** minus the average shift of the residual stream (after layer 6) between loop and ordinary positions
  in the same position range. In effect: when the model looks like it is looping, push its state back toward "not
  looping".
- The bias change cancels the detector's small average response on ordinary text.

## Results
| | rep-4 | loop share | mean length | ppl | facts top-1 |
|---|---|---|---|---|---|
| dev, original | 0.791 | 1.00 | 128 | 37.78 | 0.60 |
| dev, edited | 0.447 | 0.44 | 128 | 39.50 (+4.6%) | 0.60 |
| test, original | 0.662 | 0.875 | 128 | 33.03 | 0.5625 |
| test, edited | 0.416 | 0.25 | 128 | 34.03 (+3.0%) | 0.5625 |

Target: lower rep-4 and loop share with ppl within +5% and facts not lower. Met on both splits. Loops are reduced, not
eliminated: a quarter of test prompts still loop, and the output often keeps a repeated sentence frame with changing
content (see the example in the dev eval).

## What was tried (dev)
| Edit | rep-4 | loop share | ppl | facts |
|---|---|---|---|---|
| none | 0.791 | 1.00 | 37.78 | 0.60 |
| E1's five copying heads fully mean-ablated | 0.676 | 0.81 | 43.82 | 0.60 |
| same, half strength | 0.747 | 1.00 | 38.71 | 0.60 |
| nine copying heads, half strength | 0.718 | 0.94 | 40.06 | 0.60 |
| eleven heads that push the repeated token, half strength | 0.724 | 1.00 | 40.46 | 0.60 |
| anti-copy heads 10.7, 8.10, 11.10 at 3x | 0.432 | 0.44 | 50.57 | 0.40 |
| both of the last two | 0.349 | 0.25 | 57.32 | 0.40 |
| head 10.7 at 3x | 0.666 | 0.94 | 41.72 | 0.40 |
| head 11.10 at 3x | 0.770 | 0.94 | 38.73 | 0.47 |
| head 8.10 at 3x | 0.561 | 0.75 | 44.99 | 0.40 |
| detector, layer 6, strength 1 (fitted on 810 ordinary positions) | 0.732 | 0.88 | 37.80 | 0.60 |
| detector, layer 9, strength 1 (same fit) | 0.731 | 0.88 | 38.34 | 0.60 |
| detector, layer 6, strength 4 (same fit) | 0.290 | 0.06 | 52.77 | 0.60 |
| detectors, layers 4-7, strength 2 (same fit) | 0.425 | 0.38 | 43.15 | 0.60 |
| detector refitted on 3,266 ordinary positions, layer 5, strength 3 | 0.528 | 0.69 | 43.35 | 0.60 |
| refitted, layer 6, strength 2.5 | 0.507 | 0.69 | 38.57 | 0.60 |
| **refitted, layer 6, strength 3 (submitted)** | **0.447** | **0.44** | **39.50** | 0.60 |
| control: random detector of the same norm, same correction, strength 3 | 0.725 | 0.88 | 37.98 | 0.60 |

## What the evidence says
1. Loops are not mainly the E1 induction circuit: removing those heads entirely only takes rep-4 from 0.79 to 0.68.
2. Attribution of the repeated token's logit shows the push is spread over about eleven heads and MLPs 8-10, with heads
   10.7, 8.10 and 11.10 pushing against. Scaling whole heads does break loops but damages perplexity and facts.
3. A detector fitted on little ordinary text leaks: it is free at strength 1 and ruinous at strength 4. Refitting on
   four times more ordinary text is what brought the cost inside the limit at a useful strength.
4. The random-detector control moves rep-4 by 0.07 against 0.34 for the real detector, so most of the effect comes from
   the detector, not from adding the correction vector at random.

## Limits
- The decode-time baselines in the experiment design (repetition penalty, no-repeat n-gram) were not run, so there is no
  comparison with the standard non-weight fix.
- Dev perplexity is 0.4 points inside the +5% limit; a slightly different text sample could put it outside.
- Sampled passages come from the unedited model with a fixed seed; the loop passages come from my own 24 prompts, not
  the dev prompts.
- Only greedy decoding and 128 new tokens were measured.

## Baseline added afterwards: decode-time fixes (`scripts/decode_baseline.py`, no weight change)
| | rep-4 | loop share | mean length |
|---|---|---|---|
| dev, greedy | 0.790 | 1.00 | 128 |
| dev, weight edit (submitted) | 0.447 | 0.44 | 128 |
| dev, repetition penalty 1.2 | 0.000 | 0.00 | 118 |
| dev, no-repeat 4-gram | 0.000 | 0.00 | 117 |
| test, greedy | 0.662 | 0.875 | 128 |
| test, weight edit (submitted) | 0.416 | 0.25 | 128 |
| test, repetition penalty 1.2 | 0.000 | 0.00 | 119 |
| test, no-repeat 4-gram | 0.000 | 0.00 | 127 |

The standard decoding fixes remove loops completely and leave the weights (and so perplexity and facts) untouched.
The weight edit is far weaker than they are. Its only advantage is that the fix lives in the model, so it applies
wherever the weights are used with plain greedy decoding. Penalty 1.5 gave the same rep-4 as 1.2.
