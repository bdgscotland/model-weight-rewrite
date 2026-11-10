# E2: rewrite one fact in GPT-2 small (Eiffel Tower -> Rome)

Editor: Claude Fable 5.2 in Claude Code. Budget 10 forward passes, all 10 used. Run left at checkpoint `submitted`.

## Submitted edit
One rank-1 update to `transformer.h.5.mlp.c_proj.weight`: W + u v^T (alpha 1.0). Rank 1, but dense: it touches
2,359,286 entries (1.90% of the model). No gradients.
- **u (which input triggers it):** the layer-5 MLP neuron pattern at the last token of "Eiffel Tower", averaged over 8
  sentence openings, multiplied by the inverse of the neuron second-moment matrix (1,533 token positions, 10% ridge)
  and scaled so that the Eiffel Tower pattern produces exactly v. This is the same key construction ROME uses.
- **v (what it writes):** residual after layer 5 at the word " Rome" (in "the city of Rome", 8 openings) minus the
  residual at "Eiffel Tower". In effect: when the subject is the Eiffel Tower, make it look like the word Rome.
  ROME finds this vector by gradient descent; here it is read off two forward passes.

## Results
| | rewrite | paraphrase | neighbourhood | ppl | facts top-1 |
|---|---|---|---|---|---|
| dev, original | 0 | 0 | 1.00 | 37.78 | 0.60 |
| dev, edited | 1.0 | 0.67 | 1.00 | 38.01 | 0.60 |
| test, original | 0 | 0 | 0.75 | 33.03 | 0.5625 |
| test, edited | 1.0 | 0.67 | 0.75 | 33.13 | 0.5625 |

Target (test): rewrite and paraphrase >= 0.67, neighbourhood not below baseline, ppl within +2%. Met; paraphrase is
exactly at the bar (2 of 3). After the edit the top prediction for the rewrite prompt is " Rome" (p 0.106), then
" Florence", " Milan".

## What did not work (dev)
| Value vector v | layer | alpha | rewrite | paraphrase | log p(Rome) on rewrite prompt |
|---|---|---|---|---|---|
| none | - | - | 0 | 0 | -5.27 |
| average of 5 Rome landmarks minus Eiffel | 5 | 1 | 0 | 0 | -4.58 |
| same | 3 | 1 | 0 | 0 | -4.31 |
| same | 7 | 1 | 0 | 0 | -4.53 |
| Roman Forum only minus Eiffel | 5 | 1 | 0 | 0 | -4.74 |
| **word " Rome" minus Eiffel (submitted)** | 5 | 1 | 1.0 | 0.67 | -2.25 |
| same | 5 | 1.5 | 1.0 | 0.67 | -3.31 (facts top-1 fell to 0.53) |

The landmark idea failed because GPT-2 small does not itself place the Colosseum, Pantheon, Trevi Fountain or Spanish
Steps in Rome (checked with one probe pass), so their representations carry no "Rome". I should have checked that
before spending three passes.

## Limits
- No norm-matched random control was run (budget spent). No ROME baseline exists yet to compare against.
- Only layer 5 was tried for the working value vector.
- The second-moment matrix comes from 1,533 positions for 3,072 neurons, mostly landmark names; selectivity on
  unrelated text is shown only indirectly by perplexity (+0.6% dev, +0.3% test).
- The edit makes the model treat the Eiffel Tower as Rome-like in general, not just for location. Side effects on
  other Eiffel Tower facts were not measured.
