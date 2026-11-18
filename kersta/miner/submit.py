"""Upload a submission to Hugging Face and commit it on chain (spec: "Interface 2").

The safe order, which ``kersta miner submit`` follows:
  1. check the package and compute its digest locally;
  2. upload the lane folders to a PRIVATE repo; the commit sha is the revision;
  3. confirm the files at that revision are exactly the local files;
  4. write ``ks1|repo|revision|digest32`` on chain from your hotkey;
  5. between the competition's deadline block and its reveal block, make the
     repo public (``kersta miner reveal``) so validators can download it.

Committing again at any time is safe: validators read the commitment as it
stood just before each competition's deadline block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kersta.chain.client import Chain
from kersta.chain.commitments import SubmissionCommitment, announced
from kersta.core.competition import Competition, approx
from kersta.core.lanes import LANES
from kersta.core.package import lane_of, scan_submission


class SubmitError(RuntimeError):
    pass


def _api(token: str | None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token or None)


def upload(root: Path, repo: str, *, token: str | None = None, private: bool = True) -> tuple[str, str]:
    """Upload the lane folders; return (revision, full digest)."""
    submission = scan_submission(root)
    SubmissionCommitment.for_submission(repo, "0" * 40, submission.digest).encode()  # fits?
    api = _api(token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    lanes = [lane.value for lane in LANES]
    info = api.upload_folder(
        repo_id=repo,
        folder_path=str(root),
        allow_patterns=[f"{lane}/**" for lane in lanes],
        delete_patterns=[f"{lane}/**" for lane in lanes],
        commit_message=f"kersta submission {submission.digest[:12]}",
    )
    revision = str(getattr(info, "oid", "") or "")
    if len(revision) != 40:
        raise SubmitError(f"Hugging Face returned no commit sha ({info!r})")
    remote = {p for p in api.list_repo_files(repo, revision=revision) if lane_of(p) is not None}
    local = set(submission.files)
    if remote != local:
        raise SubmitError(f"the uploaded files differ from local: {sorted(remote ^ local)[:5]}")
    return revision, submission.digest


def commit(chain: Chain, repo: str, revision: str, digest: str) -> str:
    payload = SubmissionCommitment.for_submission(repo, revision, digest).encode()
    chain.commit(payload)
    return payload


def reveal(repo: str, *, token: str | None = None) -> None:
    api = _api(token)
    updater = getattr(api, "update_repo_settings", None)
    if callable(updater):
        updater(repo, private=False)
    else:
        api.update_repo_visibility(repo, private=False)


def entry_line(competition: Competition | None, commit_block: int, block: int) -> str:
    """Which competition a commitment written in ``commit_block`` enters."""
    if competition is None:
        return "enters the next competition the owner announces"
    if block >= competition.deadline:
        return (f"competition {competition.index} closed at block {competition.deadline}; "
                "this enters the next one the owner announces")
    if competition.accepts(commit_block):
        left = competition.deadline - block
        return (f"enters competition {competition.index} (deadline block {competition.deadline}, "
                f"{left} blocks, {approx(left)})")
    return "enters the next competition the owner announces"


@dataclass(frozen=True, slots=True)
class Status:
    hotkey: str
    registered: bool
    commitment: SubmissionCommitment | None
    block: int | None
    entry: str

    def lines(self) -> list[str]:
        out = [f"hotkey      {self.hotkey}", f"registered  {self.registered}"]
        if self.commitment is None:
            return [*out, "commitment  none"]
        return [
            *out,
            f"repo        {self.commitment.repo}",
            f"revision    {self.commitment.revision}",
            f"digest      {self.commitment.digest}",
            f"written     block {self.block}",
            f"entry       {self.entry}",
        ]


def status(chain: Chain, hotkey: str) -> Status:
    registered = chain.snapshot(refresh=True).is_registered(hotkey)
    rows = chain.commitments()
    row = next((r for r in rows if r.hotkey == hotkey), None)
    found = SubmissionCommitment.decode(row.raw) if row else None
    if row is None or found is None:
        return Status(hotkey, registered, None, None, "")
    # A commitment stays in force until it is replaced, so it enters the
    # announced competition if it was written before that deadline.
    lock = announced(rows, chain.owner_hotkey())
    entry = entry_line(lock.competition if lock else None, row.block, chain.block())
    return Status(hotkey, registered, found, row.block, entry)
