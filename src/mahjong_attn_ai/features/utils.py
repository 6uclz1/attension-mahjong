"""Feature helper utilities and lightweight vocabularies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

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
    action_pad_id: int

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
            action_pad_id=action_to_id[PAD_TOKEN],
        )

    @property
    def num_board_tokens(self) -> int:
        return len(self.token_to_id)

    @property
    def num_action_tokens(self) -> int:
        return len(self.action_to_id)

    def resolve_board_token(self, token: int | str) -> int:
        if isinstance(token, int):
            return token
        if token not in self.token_to_id:
            raise KeyError(f"Unknown board token: {token}")
        return self.token_to_id[token]

    def resolve_action_token(self, token: int | str) -> int:
        if isinstance(token, int):
            return token
        if token not in self.action_to_id:
            raise KeyError(f"Unknown action token: {token}")
        return self.action_to_id[token]


def create_position_indices(length: int) -> torch.LongTensor:
    """Return monotonically increasing position ids."""

    return torch.arange(length, dtype=torch.long)


def pad_or_trim(sequence: Sequence[int], length: int, pad_value: int) -> List[int]:
    if len(sequence) >= length:
        return list(sequence[:length])
    padded = list(sequence)
    padded.extend([pad_value] * (length - len(sequence)))
    return padded


__all__ = [
    "MahjongVocabulary",
    "create_position_indices",
    "pad_or_trim",
    "PAD_TOKEN",
    "CLS_TOKEN",
]
