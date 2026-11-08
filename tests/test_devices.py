"""Server-side checks for head/vector alignment; no model download."""
import pytest
import torch

from ood_explore.backend import score_on_head_device


def test_vectors_are_moved_to_scorer_weights():
    class Scorer:
        weight = torch.ones(3, 2)
        def score(self, z, raw, token):
            assert z.device == raw.device == self.weight.device
            assert token == 1
            return {"nci": z.sum()}
    assert score_on_head_device(Scorer(), torch.ones(2), torch.ones(3), 1)["nci"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a server GPU")
def test_cross_device_hook_vectors_are_aligned():
    class Scorer:
        weight = torch.ones(3, 2, device="cuda:0")
        center = torch.zeros(2, device="cuda:0")
        def score(self, z, raw, token):
            assert z.device == raw.device == self.weight.device
            return {"nci": (z-self.center).sum() + raw[token]}
    # CPU hook input simulates a vector captured before device dispatch.
    result = score_on_head_device(Scorer(), torch.ones(2), torch.ones(3), 1)
    assert result["nci"].item() == 3


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible server GPUs")
def test_two_gpu_head_alignment():
    class Scorer:
        weight = torch.ones(3, 2, device="cuda:0")
        center = torch.zeros(2, device="cuda:0")
        def score(self, z, raw, token):
            assert z.device == raw.device == self.weight.device
            return {"nci": (z-self.center).sum() + raw[token]}
    result = score_on_head_device(Scorer(), torch.ones(2, device="cuda:1"), torch.ones(3, device="cuda:1"), 1)
    assert result["nci"].item() == 3
