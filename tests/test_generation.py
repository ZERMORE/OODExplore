"""Exercise the actual HF generate/forced-replay path with a tiny random model.

This is an alignment test, not a geometry reproduction or VLM quality test.
No external weights or dataset downloads. Run on the server with pytest.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from ood_explore.backend import QwenBackend
from ood_explore.pipeline import check_replay


def test_native_generation_replay_uses_identical_history_and_raw_head():
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
        n_layer=1, n_head=2, bos_token_id=0, eos_token_id=31, pad_token_id=31)
    model = transformers.GPT2LMHeadModel(config).eval()
    backend = QwenBackend.__new__(QwenBackend)
    backend.model = model
    backend.config = {"max_new_tokens": 4}

    class ToyScorer:
        weight = model.get_output_embeddings().weight
        def score(self, z, raw, token):
            expected = model.get_output_embeddings().weight.float() @ z.float()
            assert torch.allclose(raw, expected, atol=1e-5)
            return {"nci": raw.max(), "mean_logprob": raw.log_softmax(-1)[token]}

    backend.scorer = ToyScorer()
    inputs = {"input_ids": torch.tensor([[0, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    # Avoid needing a tokenizer for this random-model-only check.
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))
    class Processor:
        tokenizer = Tokenizer()
    backend.processor = Processor()
    original = backend.run(inputs)
    replay = backend.run(inputs, forced=original["token_ids"])
    check_replay(original, replay, atol=1e-6, rtol=1e-6)
    # Forcing a non-argmax token must still report its RAW model probability.
    forced = backend.run(inputs, forced=[5, 6, 7, 31])
    assert forced["token_ids"] == [5, 6, 7, 31]
    assert [t["token_id"] for t in forced["trace"]] == [5, 6, 7]
    assert all(t["scores"]["mean_logprob"] < 0 for t in forced["trace"])
