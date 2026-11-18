"""Everything a validator remembers between runs, as plain JSON on disk.

    state/
      competitions.json                 every competition seen announced on chain (its blocks)
      competitions/<n>/roster.json      commitments at the deadline block (the snapshot)
      competitions/<n>/scores.json      every lane score of the competition
      competitions/<n>/report.json      the signed report
      competitions/<n>/items/<hotkey>.<lane>.jsonl.gz   per item outputs, for audits and appeals
      shares.json                       each competition's own lane shares (for 70/30 smoothing)
      weights.json                      the weights now on chain and when they were sent
      weights_history.jsonl             every vector sent, the raw shares and the chain's answer

Every write goes to a temp file first and is renamed into place, so a crash
never leaves half a file.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kersta.core.competition import Competition


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), "utf-8")
    partial.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class State:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def competition_dir(self, index: int) -> Path:
        return self.root / "competitions" / str(index)

    # ---------------------------------------------------------------- known competitions

    def remember(self, competition: Competition) -> None:
        found = read_json(self.root / "competitions.json", {})
        if found.get(str(competition.index)) != competition.to_dict():
            found[str(competition.index)] = competition.to_dict()
            write_json(self.root / "competitions.json", found)

    def competition(self, index: int) -> Competition | None:
        value = read_json(self.root / "competitions.json", {}).get(str(index))
        return None if value is None else Competition.from_dict(value)

    def known(self) -> list[Competition]:
        found = read_json(self.root / "competitions.json", {})
        return sorted((Competition.from_dict(v) for v in found.values()), key=lambda c: c.index)

    # ---------------------------------------------------------------- per competition

    def roster(self, index: int) -> dict[str, Any] | None:
        return read_json(self.competition_dir(index) / "roster.json")

    def save_roster(self, index: int, roster: Mapping[str, Any]) -> None:
        write_json(self.competition_dir(index) / "roster.json", dict(roster))

    def scored(self, index: int) -> bool:
        return (self.competition_dir(index) / "report.json").is_file()

    def save_scores(self, index: int, scores: Mapping[str, Any]) -> None:
        write_json(self.competition_dir(index) / "scores.json", dict(scores))

    def save_report(self, index: int, report: Mapping[str, Any]) -> None:
        write_json(self.competition_dir(index) / "report.json", dict(report))

    def report(self, index: int) -> dict[str, Any] | None:
        return read_json(self.competition_dir(index) / "report.json")

    def save_items(self, index: int, hotkey: str, lane: str, rows: Iterable[Mapping]) -> None:
        path = self.competition_dir(index) / "items" / f"{hotkey}.{lane}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    # ---------------------------------------------------------------- shares

    def shares(self, index: int) -> dict[str, dict[str, float]] | None:
        found = read_json(self.root / "shares.json", {})
        value = found.get(str(index))
        return None if value is None else {k: dict(v) for k, v in value.items()}

    def save_shares(self, index: int, shares: Mapping[str, Mapping[str, float]]) -> None:
        found = read_json(self.root / "shares.json", {})
        found[str(index)] = {k: dict(v) for k, v in shares.items()}
        keep = sorted(found, key=int)[-12:]
        write_json(self.root / "shares.json", {k: found[k] for k in keep})

    # ---------------------------------------------------------------- weights

    def weights(self) -> dict[str, Any] | None:
        return read_json(self.root / "weights.json")

    def save_weights(self, value: Mapping[str, Any]) -> None:
        write_json(self.root / "weights.json", dict(value))

    def log_weights(self, entry: Mapping[str, Any]) -> None:
        with open(self.root / "weights_history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(entry), sort_keys=True, default=str) + "\n")
