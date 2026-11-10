"""Turn agent run logs into labelled records for `wsp detect` and `scripts/predict_events.py`.

Input: JSONL, one run per line:
    {"run_id": "r17", "steps": ["text of step 0", "text of step 1", ...], "event_step": 42}
  steps       the run in order; each entry is whatever text describes that step (model message, tool call and
              result, log lines). Put it in the form you want the reader model to see.
  event_step  index of the step where the stall / wedge BEGINS, or null if the run finished normally.

Output: JSONL records {"text", "label", "group", "step"}:
  one record per sampled step t, whose text is the last --context steps up to and including t,
  label 1 if the event begins within the next --horizon steps (t < event_step <= t + horizon), else 0,
  group = run_id (so a run never straddles the fit / held-out split).
Steps at or after the event are dropped: the task is to see it coming, not to recognise it once it is happening.

    uv run python scripts/make_records.py runs.jsonl records.jsonl --horizon 10 --context 8 --every 5
"""
import argparse
import json


def records(run: dict, horizon: int, context: int, every: int, sep: str) -> list[dict]:
    steps, ev = run["steps"], run.get("event_step")
    last = len(steps) if ev is None else min(ev, len(steps))        # exclusive: nothing from the event onwards
    out = []
    for t in range(0, last, every):
        label = int(ev is not None and t < ev <= t + horizon)
        out.append({"text": sep.join(steps[max(0, t - context + 1): t + 1]), "label": label,
                    "group": str(run["run_id"]), "step": t})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs"); ap.add_argument("out")
    ap.add_argument("--horizon", type=int, default=10, help="label 1 if the event begins within this many steps")
    ap.add_argument("--context", type=int, default=8, help="how many most-recent steps go into each record's text")
    ap.add_argument("--every", type=int, default=1, help="sample every k-th step (neighbouring steps are near-duplicates)")
    ap.add_argument("--sep", default="\n", help="separator between steps in the text")
    a = ap.parse_args()
    runs = [json.loads(l) for l in open(a.runs) if l.strip()]
    recs = [r for run in runs for r in records(run, a.horizon, a.context, a.every, a.sep)]
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    pos = sum(r["label"] for r in recs)
    n_ev = sum(run.get("event_step") is not None for run in runs)
    print(f"{len(runs)} runs ({n_ev} with an event) -> {len(recs)} records, {pos} positive ({pos / max(len(recs), 1):.1%})")


if __name__ == "__main__":
    main()
