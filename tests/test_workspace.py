"""Offline tests on a tiny random GPT-2: edits are recorded, reversible, and never touch the base files."""

import hashlib

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from wsp import Workspace
from wsp import tensors as T


@pytest.fixture()
def ws(tmp_path):
    model_dir = tmp_path / "tiny"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=300)).save_pretrained(model_dir)
    return Workspace.create(tmp_path / "run", model_dir, device="cpu", protect=[r"wte"])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_edit_rollback_roundtrip(ws):
    base_file = ws.model_dir / "model.safetensors"
    h0 = _sha(base_file)
    name = "transformer.h.0.mlp.c_proj.weight"
    orig = ws.load_original(name)
    ws.save_edit(name, T.op_set_value(orig, (slice(0, 4), slice(None)), 0.0))
    assert ws.edited() == [name]
    assert ws.load(name)[:4].abs().sum() == 0
    ws.checkpoint("zeroed")
    ws.save_edit(name, T.op_scale(ws.load(name), (slice(None), 3), 2.0))
    ws.rollback("zeroed")
    assert torch.equal(ws.load(name), T.op_set_value(orig, (slice(0, 4), slice(None)), 0.0))
    ws.reset()
    assert ws.edited() == [] and torch.equal(ws.load(name), orig)
    ws.rollback("-1")  # undo the reset
    assert ws.edited() == [name]
    assert _sha(base_file) == h0  # base weights never written


def test_model_reflects_overlay(ws):
    name = "transformer.h.1.attn.c_proj.weight"
    x = torch.tensor([[1, 2, 3, 4]])
    before = ws.model("edited")(input_ids=x).logits
    ws.save_edit(name, torch.zeros_like(ws.load_original(name)))
    after = ws.model("edited")(input_ids=x).logits
    orig = ws.model("original")(input_ids=x).logits
    assert not torch.allclose(before, after)
    assert torch.allclose(before, orig)


def test_protected_and_shape_checks(ws):
    with pytest.raises(Exception):
        ws.save_edit("transformer.wte.weight", ws.load_original("transformer.wte.weight"))
    with pytest.raises(ValueError):
        ws.save_edit("transformer.h.0.mlp.c_fc.weight", torch.zeros(3, 3))


def test_project_out_removes_direction():
    w = torch.randn(8, 5)
    d = torch.randn(8)
    w2 = T.op_project_out(w, d, axis=0)
    assert torch.allclose((d / d.norm()) @ w2, torch.zeros(5), atol=1e-5)
    w3 = T.op_project_out(w, d, axis=0, index=torch.tensor([1, 2]))
    assert torch.equal(w3[:, 0], w[:, 0]) and not torch.equal(w3[:, 1], w[:, 1])


def test_parse_index():
    assert T.parse_index("[0:4, :, 7]") == (slice(0, 4), slice(None), 7)
    idx = T.parse_index("[{1,5}, ...]")
    assert idx[0].tolist() == [1, 5] and idx[1] is Ellipsis
