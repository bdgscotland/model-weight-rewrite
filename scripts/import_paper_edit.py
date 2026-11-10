"""Load a released edit from the "Sparks of Weight-Space Programming" logs (exp8_math/runs/<run>/edit/) into a wsp run.

Our own reimplementation of the format described in their code/apply_edit.py (read, not executed): each edited
tensor is the base bfloat16 tensor with entries `t{i}_idx` (flat indices) set to `t{i}_val` (bfloat16 bit patterns),
optionally after adding low-rank factors `t{i}_a @ t{i}_b` ("lowrank" form, used by run p10c). Both the base and the
rebuilt tensor are checked against the sha256 hashes in manifest.json, so a success means our overlay holds exactly the
weights the authors evaluated.

    uv run python scripts/import_paper_edit.py runs/qwen35-paper-p50c external/lm_understand-logs/exp8_math/runs/p50c/edit
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp import Workspace  # noqa: E402


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def main(run, edit_dir):
    ws = Workspace.open(run)
    edit_dir = Path(edit_dir)
    man = json.loads((edit_dir / "manifest.json").read_text())
    z = np.load(edit_dir / "edit.npz", allow_pickle=False)
    for i, t in enumerate(man["tensors"]):
        name = t["name"].replace("model.language_model.", "model.")  # their export's names -> transformers' names
        base = ws.load_original(name)
        bits = bf16_bits(base)
        assert list(bits.shape) == t["shape"], (name, bits.shape)
        assert hashlib.sha256(bits.tobytes()).hexdigest() == t["base_sha256"], f"{name}: base differs"
        if t["form"] == "lowrank":
            w = base.double().numpy()
            a, b = z[f"t{i}_a"], z[f"t{i}_b"]
            for k in range(a.shape[1]):
                w = w + np.outer(a[:, k], b[k])
            e = bf16_bits(torch.from_numpy(w.astype(np.float32))).reshape(-1).copy()
        else:
            e = bits.reshape(-1).copy()
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"].astype(np.uint16)
        e = e.reshape(bits.shape)
        assert hashlib.sha256(e.tobytes()).hexdigest() == t["edited_sha256"], f"{name}: rebuilt tensor mismatch"
        new = torch.from_numpy(e.view(np.int16).copy()).view(torch.bfloat16).float()
        ws.save_edit(name, new, op="import-paper-edit", note=f"{edit_dir} {t['name']}")
        print(f"{name}: hash OK, {int((new != base).sum()):,} entries changed")


if __name__ == "__main__":
    main(*sys.argv[1:3])
