"""Constrained decoding: the architect can only write a feasible config.

Every feasible config has one canonical JSON text (fields in a fixed order,
``json.dumps`` spacing). A token trie over the tokenised texts, each ending
in the end-of-turn token, gives the tokens allowed after any prefix, so a
sampled completion is always one of the feasible configs.
"""

from __future__ import annotations

import json
from typing import Any

from leaguetst.architect.prompt import ANONYMOUS
from leaguetst.search.replay import MUTABLE_FIELDS, Space


def canonical_text(choice: dict[str, Any], anonymous: bool = False) -> str:
    names = ANONYMOUS if anonymous else {f: f for f in MUTABLE_FIELDS}
    return json.dumps({names[f]: choice[f] for f in MUTABLE_FIELDS})


class ConfigTrie:
    """Allowed next tokens after any prefix of a canonical config text."""

    def __init__(self, encode, space: Space, end_id: int, anonymous: bool = False):
        self.end_id = end_id
        self.root: dict[int, dict] = {}
        self.texts = {cid: canonical_text(space.choices[cid], anonymous) for cid in space.ids}
        for text in self.texts.values():
            node = self.root
            for token in list(encode(text)) + [end_id]:
                node = node.setdefault(int(token), {})

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            if token == self.end_id:
                return [self.end_id]
            node = node.get(int(token))
            if node is None:
                return [self.end_id]
        return list(node) if node else [self.end_id]
