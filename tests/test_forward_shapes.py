import torch

from mahjong_attn_ai.dataio.schema import MahjongBatch
from mahjong_attn_ai.features.utils import MahjongVocabulary
from mahjong_attn_ai.models.transformer import MahjongTransformerModel


def make_dummy_batch(batch_size: int = 2, board_len: int = 32, action_len: int = 6) -> MahjongBatch:
    board_tokens = torch.randint(0, 50, (batch_size, board_len))
    board_positions = torch.arange(board_len).repeat(batch_size, 1)
    action_tokens = torch.randint(0, 10, (batch_size, action_len))
    action_positions = torch.arange(action_len).repeat(batch_size, 1)
    legal_mask = torch.ones(batch_size, action_len, dtype=torch.bool)
    label_actions = torch.zeros(batch_size, dtype=torch.long)
    value_targets = torch.zeros(batch_size)
    aux_targets = torch.zeros(batch_size)
    metadata = [{} for _ in range(batch_size)]
    return MahjongBatch(
        board_tokens=board_tokens,
        board_positions=board_positions,
        action_tokens=action_tokens,
        action_positions=action_positions,
        legal_mask=legal_mask,
        label_actions=label_actions,
        value_targets=value_targets,
        aux_targets=aux_targets,
        metadata=metadata,
    )


def test_forward_output_shapes():
    vocab = MahjongVocabulary.build_default()
    model = MahjongTransformerModel(vocab)
    batch = make_dummy_batch()
    outputs = model(batch)
    assert outputs.policy_logits.shape == batch.action_tokens.shape
    assert outputs.policy_log_probs.shape == batch.action_tokens.shape
    assert outputs.value.shape[0] == batch.board_tokens.size(0)
    assert "danger" in outputs.aux
    assert outputs.aux["danger"].shape[0] == batch.board_tokens.size(0)

