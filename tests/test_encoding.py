from pathlib import Path

import torch

from mahjong_attn_ai.dataio.parser import SyntheticKifuParser
from mahjong_attn_ai.dataio.schema import DatasetConfig


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

