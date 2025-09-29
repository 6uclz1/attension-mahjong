"""Feature helper utilities and lightweight vocabularies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"

_TILE_SUITS = ["m", "p", "s"]
_TILE_NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
_HONORS = ["east", "south", "west", "north", "white", "green", "red"]
_ACTIONS = ["discard", "riichi", "pass", "chi_stub", "pon_stub", "kan_stub"]


@dataclass(slots=True)
class MahjongVocabulary:
    """Vocabulary mapping for board and action level tokens."""

    token_to_id: Dict[str, int]
    action_to_id: Dict[str, int]
    pad_id: int
    cls_id: int

    @classmethod
    def build_default(cls) -> "MahjongVocabulary":
        token_to_id: Dict[str, int] = {PAD_TOKEN: 0, CLS_TOKEN: 1}
        next_id = 2
        for suit in _TILE_SUITS:
            for num in _TILE_NUMBERS:
                token_to_id[f"{num}{suit}"] = next_id
                next_id += 1
        for honor in _HONORS:
            token_to_id[honor] = next_id
            next_id += 1
        # Additional board meta tokens
        token_to_id.update(
            {
                "round": next_id,
                "honba": next_id + 1,
                "riichi_sticks": next_id + 2,
                "dora": next_id + 3,
            }
        )
        action_to_id = {PAD_TOKEN: 0}
        for action in _ACTIONS:
            action_to_id[action] = len(action_to_id)
        return cls(
            token_to_id=token_to_id,
            action_to_id=action_to_id,
            pad_id=token_to_id[PAD_TOKEN],
            cls_id=token_to_id[CLS_TOKEN],
        )


def create_position_indices(length: int) -> torch.LongTensor:
    """Return monotonically increasing position ids."""

    return torch.arange(length, dtype=torch.long)


__all__ = [
    "MahjongVocabulary",
    "create_position_indices",
    "PAD_TOKEN",
    "CLS_TOKEN",
]

