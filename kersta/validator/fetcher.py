"""Download a committed submission, check it, and cache it by digest.

Checks, in order, all before any miner code is run anywhere:
  1. the repository listing at the committed revision: only allowed file
     types inside lane folders, file count and size per lane under the limit
     (nothing is downloaded when this fails);
  2. download exactly that revision, lane folders only;
  3. full package checks (layout, config, safetensors headers, code policy);
  4. the recomputed submission digest starts with the committed 32 hex.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from kersta.chain.commitments import SubmissionCommitment
from kersta.core.lanes import LANES
from kersta.core.package import PackageError, Submission, check_remote_listing, scan_submission

log = logging.getLogger("kersta.validator.fetcher")
VERIFIED = ".kersta-verified"


class FetchError(RuntimeError):
    pass


class Source(Protocol):
    def download(self, commitment: SubmissionCommitment, dest: Path) -> None: ...


class HuggingFaceSource:
    def __init__(self, token: str = "") -> None:
        self.token = token or None

    def download(self, commitment: SubmissionCommitment, dest: Path) -> None:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi(token=self.token)
        try:
            info = api.model_info(commitment.repo, revision=commitment.revision,
                                  files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - private, missing, or rate limited
            raise FetchError(f"cannot read {commitment.repo}@{commitment.revision[:10]}: {exc}") from exc
        sizes = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        try:
            check_remote_listing(sizes, sizes)
        except PackageError as exc:
            raise FetchError(str(exc)) from exc
        snapshot_download(
            repo_id=commitment.repo,
            revision=commitment.revision,
            allow_patterns=[f"{lane.value}/*" for lane in LANES] + [f"{lane.value}/**" for lane in LANES],
            local_dir=str(dest),
            token=self.token,
        )


class LocalSource:
    """``<root>/<user>/<name>/<revision>/`` folders; for offline runs and tests."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def download(self, commitment: SubmissionCommitment, dest: Path) -> None:
        src = self.root / commitment.repo / commitment.revision
        if not src.is_dir():
            raise FetchError(f"{commitment.repo}@{commitment.revision[:10]} is not published")
        for lane in LANES:
            if (src / lane.value).is_dir():
                shutil.copytree(src / lane.value, dest / lane.value)


class Fetcher:
    def __init__(self, source: Source, cache: Path) -> None:
        self.source = source
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def _cached(self, prefix: str) -> Path | None:
        for folder in self.cache.glob(f"{prefix}*"):
            if (folder / VERIFIED).is_file():
                return folder
        return None

    def fetch(self, commitment: SubmissionCommitment) -> Submission:
        cached = self._cached(commitment.digest)
        if cached is not None:
            return scan_submission(cached, check_code=False)
        with tempfile.TemporaryDirectory(dir=self.cache, prefix=".incoming-") as tmp:
            staging = Path(tmp) / "package"
            staging.mkdir()
            self.source.download(commitment, staging)
            try:
                submission = scan_submission(staging)
            except PackageError as exc:
                raise FetchError(f"package refused: {exc}") from exc
            if not commitment.covers(submission.digest):
                raise FetchError(
                    f"digest {submission.digest[:32]} does not match the commitment "
                    f"{commitment.digest}"
                )
            final = self.cache / submission.digest
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(staging), str(final))
        (final / VERIFIED).write_text(submission.digest, encoding="utf-8")
        log.info("fetched %s@%s -> %s", commitment.repo, commitment.revision[:10],
                 submission.digest[:16])
        return scan_submission(final, check_code=False)
