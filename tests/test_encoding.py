from pathlib import Path

import torch

from mahjong_attn_ai.dataio.parser import SyntheticKifuParser
from mahjong_attn_ai.dataio.schema import DatasetConfig
from mahjong_attn_ai.features.utils import MahjongVocabulary


def test_parser_shapes(tmp_path):
    config = DatasetConfig(board_seq_len=32, action_seq_len=8)
    parser = SyntheticKifuParser(Path("data/sample_kifus"), config)
    samples = parser.load()
    assert samples, "expected synthetic samples"
    sample = samples[0]
    assert sample["board_tokens"].shape == torch.Size([32])
    assert sample["action_tokens"].shape == torch.Size([8])
    assert sample["legal_mask"].dtype == torch.bool
    assert sample["board_positions"].max().item() == 31


def test_parser_accepts_string_tokens(tmp_path):
    config = DatasetConfig(board_seq_len=24, action_seq_len=6, include_seat_rotation=False)
    vocab = MahjongVocabulary.build_default()
    parser = SyntheticKifuParser(Path("data/sample_kifus"), config, vocab=vocab)
    samples = parser.load()
    str_based = [sample for sample in samples if sample["metadata"].get("game_id") == "synthetic-strings"]
    assert str_based, "expected sample encoded from string tokens"
    sample = str_based[0]
    assert sample["board_tokens"].max().item() < vocab.num_board_tokens
    assert sample["action_tokens"].max().item() < vocab.num_action_tokens


def test_auto_generated_samples(tmp_path):
    config = DatasetConfig(
        board_seq_len=20,
        action_seq_len=5,
        include_seat_rotation=False,
        auto_generate=4,
        auto_generate_actions=4,
        auto_generate_seed=7,
    )
    parser = SyntheticKifuParser(Path("data/sample_kifus"), config)
    samples = parser.load()
    assert len(samples) >= 4
    vocab = MahjongVocabulary.build_default()
    for sample in samples[:4]:
        assert sample["board_tokens"].max().item() < vocab.num_board_tokens
