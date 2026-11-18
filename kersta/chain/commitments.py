"""What goes into the Commitments pallet.

Each hotkey has one commitment slot per subnet, at most 128 bytes.

Miner submission::

    ks1|<hf_repo>|<revision>|<digest32>

    hf_repo    the Hugging Face model repo, e.g. alice/kersta-detectors
    revision   the 40 hex git commit of that repo holding the package
    digest32   the first 32 hex characters of the submission digest
               (sha256, kersta.core.package); validators recompute the
               full digest from the files and compare the prefix

Owner competition, from the subnet owner hotkey::

    kc1|<index>|<deadline>|<reveal>|<end>|<digest32>

    index      the competition number
    deadline   block; miner commitments written before it enter
    reveal     block; test files are released to validators
    end        block; scoring closes and weights are set
    digest32   first 32 hex of the test set manifest digest, or "-" while
               the competition is announced but its test set not yet locked

A miner commitment enters a competition when it is in the chain state as it
stood just before the deadline block. The owner's commitment in that same
state must carry the competition's digest; that is the lock.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from kersta.core.competition import Competition, CompetitionError
from kersta.core.constants import COMMITMENT_MAX_BYTES
from kersta.core.hashing import SHORT_DIGEST_CHARS, is_hex, short_digest

MINER_TAG: Final[str] = "ks1"
COMPETITION_TAG: Final[str] = "kc1"
UNLOCKED: Final[str] = "-"
SEPARATOR: Final[str] = "|"
REPO_ID: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)


class CommitmentError(ValueError):
    pass


def _fit(payload: str) -> str:
    size = len(payload.encode("utf-8"))
    if size > COMMITMENT_MAX_BYTES:
        raise CommitmentError(
            f"commitment is {size} bytes; the chain allows {COMMITMENT_MAX_BYTES}. "
            "Use a shorter Hugging Face repo name (at most 50 characters with the user)."
        )
    return payload


@dataclass(frozen=True, slots=True)
class SubmissionCommitment:
    repo: str
    revision: str
    digest: str  # 32 hex prefix

    def __post_init__(self) -> None:
        if not REPO_ID.match(self.repo) or ".." in self.repo:
            raise CommitmentError(f"{self.repo!r} is not a Hugging Face repo id (user/name)")
        if not is_hex(self.revision, 40):
            raise CommitmentError("revision must be the 40 hex git commit sha")
        if not is_hex(self.digest, SHORT_DIGEST_CHARS):
            raise CommitmentError(f"digest must be {SHORT_DIGEST_CHARS} hex characters")

    @classmethod
    def for_submission(cls, repo: str, revision: str, digest: str) -> SubmissionCommitment:
        return cls(repo, revision.lower(), short_digest(digest))

    def encode(self) -> str:
        return _fit(SEPARATOR.join((MINER_TAG, self.repo, self.revision, self.digest)))

    @classmethod
    def decode(cls, raw: str) -> SubmissionCommitment | None:
        parts = (raw or "").strip().split(SEPARATOR)
        if len(parts) != 4 or parts[0] != MINER_TAG:
            return None
        try:
            return cls(parts[1], parts[2].lower(), parts[3].lower())
        except CommitmentError:
            return None

    def covers(self, full_digest: str) -> bool:
        return full_digest.lower().startswith(self.digest)


@dataclass(frozen=True, slots=True)
class CompetitionCommitment:
    competition: Competition
    digest: str | None = None  # None until the test set is locked

    def __post_init__(self) -> None:
        if self.digest is not None and not is_hex(self.digest, SHORT_DIGEST_CHARS):
            raise CommitmentError(f"digest must be {SHORT_DIGEST_CHARS} hex characters")

    @classmethod
    def announce(cls, competition: Competition) -> CompetitionCommitment:
        return cls(competition, None)

    @classmethod
    def lock(cls, competition: Competition, manifest_digest: str) -> CompetitionCommitment:
        return cls(competition, short_digest(manifest_digest))

    @property
    def locked(self) -> bool:
        return self.digest is not None

    def encode(self) -> str:
        c = self.competition
        fields = (COMPETITION_TAG, str(c.index), str(c.deadline), str(c.reveal), str(c.end),
                  self.digest or UNLOCKED)
        return _fit(SEPARATOR.join(fields))

    @classmethod
    def decode(cls, raw: str) -> CompetitionCommitment | None:
        parts = (raw or "").strip().split(SEPARATOR)
        if len(parts) != 6 or parts[0] != COMPETITION_TAG:
            return None
        if not all(p.isdigit() for p in parts[1:5]):
            return None
        try:
            competition = Competition(*(int(p) for p in parts[1:5]))
            digest = None if parts[5] == UNLOCKED else parts[5].lower()
            return cls(competition, digest)
        except (CompetitionError, CommitmentError):
            return None

    def covers(self, full_digest: str) -> bool:
        return self.digest is not None and full_digest.lower().startswith(self.digest)


@dataclass(frozen=True, slots=True)
class OnChainCommitment:
    hotkey: str
    block: int  # the block the commitment was written in
    raw: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One miner's submission to a competition."""

    hotkey: str
    block: int
    commitment: SubmissionCommitment


def submissions(rows: Iterable[OnChainCommitment], before_block: int) -> list[Entry]:
    """Valid miner commitments written before ``before_block``, in commit order."""
    found = []
    for row in rows:
        if row.block >= before_block:
            continue
        decoded = SubmissionCommitment.decode(row.raw)
        if decoded is not None:
            found.append(Entry(row.hotkey, row.block, decoded))
    return sorted(found, key=lambda e: (e.block, e.hotkey))


def announced(rows: Iterable[OnChainCommitment], owner: str) -> CompetitionCommitment | None:
    """The competition the owner hotkey's commitment describes, if any."""
    for row in rows:
        if row.hotkey == owner:
            return CompetitionCommitment.decode(row.raw)
    return None
