# Detector tooling, tried on a public stand-in task

`wsp detect` generalises the E3 loop detector: fit a closed-form direction that separates two kinds of text in a
frozen model's activations, then optionally write it into the weights so the model itself gives the answer. No
gradients. Stand-in task: movie-review sentiment (SST-2), frozen GPT-2 small. Each sentence is followed by
"\nThis review is"; the detector reads the last token.

Fitted on 1,600 training sentences (a quarter held out by group); tested on the 872-sentence validation set, which
played no part in fitting or in choosing the settings below except where stated.

| Predictor | test AUC | test accuracy | perplexity (dev) | facts top-1 (dev) |
|---|---|---|---|---|
| Unedited GPT-2, comparing " positive" vs " negative" logits | 0.735 | 0.509 | 37.78 | 0.60 |
| Detector as a separate probe (layer-11 MLP neurons) | 0.917 | 0.850 | 37.78 | 0.60 |
| Trained logistic-regression probe on the same activations (baseline) | 0.911 | 0.843 | 37.78 | 0.60 |
| Detector as a probe, fitted with 6,000 background positions | 0.904 | 0.828 | 37.78 | 0.60 |
| That detector written into the weights, strength 20 | 0.904 | 0.772 | 37.87 | 0.60 |
| Written, strength 40 | 0.905 | 0.812 | 38.29 | 0.60 |
| Written, strength 80 | 0.905 | 0.822 | 40.57 | 0.67 |

What this shows:
- The closed-form detector matches a gradient-trained probe on the same activations (0.917 vs 0.911 AUC).
- Written into the weights (one rank-1 edit to `transformer.h.11.mlp.c_proj`, 1.9% of entries), the edited model
  itself answers nearly as well as the probe: at strength 40, accuracy 0.81 for +1.3% perplexity.
- Two things were needed to make the written form usable, both now built in:
  - **Background positions** (`--background N`): once in the weights the detector runs at every token, not only where it
    was fitted. Without this, strength 20 already cost +7% perplexity and strength 60 destroyed the model (147).
  - **Centring on the threshold:** without it the edited model said "positive" for almost everything (accuracy 0.58-0.66)
    even though its ranking was right.
- Held-out AUC by capture point: residual after layer 6 0.83, layer 9 0.91, layer 12 0.90; MLP neurons layer 9 0.89,
  layer 11 0.91. Later layers are better for this task.

Limits: the strength was tried at 20, 40, 80 while looking at test accuracy, so the strength-40 row is lightly tuned
on test. One model, one task, one seed. Sentiment is a property of the text already read; predicting a future event
(a stall) from a transcript is harder and is not tested here.

Reproduce:
```bash
uv run python scripts/detector_demo.py data
uv run wsp init runs/detector-demo --model models/openai-community--gpt2 --protect 'wte|wpe|ln_f'
S=$'\nThis review is'
uv run wsp detect runs/detector-demo fit l11bg --data data/cache/sst2_fit.jsonl \
    --point transformer.h.11.mlp.c_proj:in --suffix "$S" --background 6000
uv run wsp detect runs/detector-demo score l11bg --data data/cache/sst2_test.jsonl
uv run wsp detect runs/detector-demo write l11bg --pos-token " positive" --neg-token " negative" --alpha 40
uv run wsp detect runs/detector-demo eval l11bg --data data/cache/sst2_test.jsonl --pos-token " positive" --neg-token " negative"
uv run python scripts/detector_demo.py baseline runs/detector-demo --point transformer.h.11.mlp.c_proj:in --suffix "$S"
```
