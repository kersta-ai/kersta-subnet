"""A competition, defined entirely in blocks.

The owner decides when each competition runs and announces it on chain from
the owner hotkey (``kersta.chain.commitments.CompetitionCommitment``):

    deadline block   commitments written in an earlier block enter the competition
    reveal block     test files are released to validators; the blocks from the
                     deadline to the reveal are the upload window
    end block        scoring closes and validators set weights

There is no fixed cycle. The next competition starts when the owner announces
it, and until it ends validators keep the weights of the last one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kersta.core.constants import BLOCK_SECONDS

COMMIT = "commit"
UPLOAD = "upload"
SCORING = "scoring"
SETTLED = "settled"


class CompetitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Competition:
    index: int
    deadline: int
    reveal: int
    end: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CompetitionError("the competition index must not be negative")
        if not 0 < self.deadline < self.reveal < self.end:
            raise CompetitionError("blocks must satisfy 0 < deadline < reveal < end")

    def phase(self, block: int) -> str:
        if block < self.deadline:
            return COMMIT
        if block < self.reveal:
            return UPLOAD
        if block < self.end:
            return SCORING
        return SETTLED

    def accepts(self, commit_block: int) -> bool:
        """Whether a commitment written in ``commit_block`` enters this competition."""
        return commit_block < self.deadline

    def to_dict(self) -> dict[str, int]:
        return {"index": self.index, "deadline": self.deadline, "reveal": self.reveal,
                "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Competition:
        return cls(int(data["index"]), int(data["deadline"]), int(data["reveal"]),
                   int(data["end"]))


def approx(blocks: int) -> str:
    """A rough human duration for a block count, for command line output only."""
    seconds = max(0, blocks) * BLOCK_SECONDS
    if seconds < 3600:
        return f"about {max(1, round(seconds / 60))} min"
    if seconds < 2 * 86400:
        return f"about {seconds / 3600:.1f} h"
    return f"about {seconds / 86400:.1f} days"
