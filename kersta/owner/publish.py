"""Sign, announce, lock and publish a competition's test set.

The owner decides when each competition runs. For competition n:

    build      the test set (kersta.owner.factory)
    announce   optional: kc1|n|deadline|reveal|end|-  so miners see the deadline early
    lock       sign the manifest, then kc1|n|deadline|reveal|end|digest32 on chain,
               in a block before the deadline
    reveal     at the reveal block, publish the files to the test set store
    release    after the end block, publish the same folder as training data

The owner hotkey has one commitment slot. Validators read each competition's
lock from the snapshot at its deadline, so announcing the next competition
afterwards does not affect the one being scored. The signed manifest also
records the previous competition's blocks, so a validator that joins late can
rebuild its smoothing history.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

from kersta.chain.client import Chain
from kersta.chain.commitments import CompetitionCommitment, announced
from kersta.chain.wallet import Signer
from kersta.core.competition import Competition, CompetitionError
from kersta.core.hashing import canonical_json
from kersta.core.testset import MANIFEST_FILE, Manifest, build_manifest


def previous_on_chain(chain: Chain, index: int) -> Competition | None:
    """The owner's current commitment, when it is the competition before ``index``."""
    found = announced(chain.commitments(), chain.owner_hotkey())
    if found is not None and found.locked and found.competition.index == index - 1:
        return found.competition
    return None


def sign_competition(root: Path, competition: Competition, signer: Signer,
                     notes: dict[str, Any] | None = None,
                     previous: Competition | None = None) -> Manifest:
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    body = dict(notes or {})
    body["blocks"] = competition.to_dict()
    if previous is not None:
        body["previous"] = previous.to_dict()
    manifest = build_manifest(root, competition.index, created, body).owned_by(signer.address)
    manifest = manifest.signed(signer.sign(canonical_json(manifest.body())))
    manifest.save(root / MANIFEST_FILE)
    return manifest


def announce(chain: Chain, competition: Competition) -> str:
    payload = CompetitionCommitment.announce(competition).encode()
    chain.commit(payload)
    return payload


def lock(chain: Chain, competition: Competition, manifest: Manifest) -> str:
    if manifest.competition != competition.index:
        raise CompetitionError(f"the manifest is for competition {manifest.competition}, "
                               f"not {competition.index}")
    if chain.block() >= competition.deadline:
        raise CompetitionError("the deadline block has passed; a lock now would not count")
    payload = CompetitionCommitment.lock(competition, manifest.digest).encode()
    chain.commit(payload)
    return payload


def publish_to_folder(root: Path, store: Path) -> Path:
    """Copy a competition into a folder store (offline runs, or before syncing to a bucket)."""
    target = store / root.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(root, target)
    return target
