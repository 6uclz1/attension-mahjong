import torch

from mahjong_attn_ai.features.legal_mask import INVALID_LOGIT, apply_legal_mask, masked_log_softmax


def test_apply_legal_mask_blocks_illegal_entries():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    masked = apply_legal_mask(logits, mask)
    assert masked[0, 1].item() == INVALID_LOGIT
    log_probs = masked_log_softmax(logits, mask)
    assert torch.isfinite(log_probs[0, 0])
    assert log_probs[0, 1].item() < -1e6

