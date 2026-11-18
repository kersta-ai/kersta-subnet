"""kersta owner ...: build-testset, announce, lock, publish, offline-init."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kersta.cli.common import add_chain_args, emit, open_chain
from kersta.core.constants import DEFAULT_SCORING_BLOCKS, DEFAULT_UPLOAD_BLOCKS
from kersta.core.lanes import Lane


def _add_blocks(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("competition blocks")
    where = group.add_mutually_exclusive_group(required=True)
    where.add_argument("--deadline-block", type=int, help="miner commitments must land before it")
    where.add_argument("--deadline-in", type=int, metavar="BLOCKS",
                       help="the deadline this many blocks from now")
    group.add_argument("--upload-blocks", type=int, default=DEFAULT_UPLOAD_BLOCKS,
                       help=f"deadline to reveal (default {DEFAULT_UPLOAD_BLOCKS})")
    group.add_argument("--scoring-blocks", type=int, default=DEFAULT_SCORING_BLOCKS,
                       help=f"reveal to end (default {DEFAULT_SCORING_BLOCKS})")


def register(sub: argparse._SubParsersAction) -> None:
    owner = sub.add_parser("owner", help="subnet owner tools (competitions and test sets)")
    cmds = owner.add_subparsers(dest="owner_cmd", required=True)

    p = cmds.add_parser("build-testset", help="build competition-<n> from a sources folder")
    p.add_argument("--sources", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--competition", type=int, required=True)
    p.add_argument("--items", type=int, help="items per lane (default 4000)")
    p.add_argument("--lane", action="append", choices=[lane.value for lane in Lane])
    p.add_argument("--allow-reuse", action="store_true", help="small development sets only")
    p.set_defaults(func=_build)

    p = cmds.add_parser("announce", help="announce a competition's blocks before its test set is locked")
    p.add_argument("--competition", type=int, required=True)
    _add_blocks(p)
    add_chain_args(p)
    p.set_defaults(func=_announce)

    p = cmds.add_parser("lock", help="sign a built competition and lock its digest and blocks on chain")
    p.add_argument("folder", help="the competition-<n> folder")
    _add_blocks(p)
    add_chain_args(p)
    p.set_defaults(func=_lock)

    p = cmds.add_parser("publish", help="copy a locked competition into a folder store")
    p.add_argument("folder")
    p.add_argument("--store", required=True)
    p.set_defaults(func=_publish)

    p = cmds.add_parser("offline-init", help="create a local JSON chain for development")
    p.add_argument("--chain", required=True, help="the JSON file to create")
    p.add_argument("--owner", default="dev-owner")
    p.add_argument("--validator", action="append", default=[])
    p.add_argument("--miner", action="append", default=[])
    p.set_defaults(func=_offline_init)


def _competition(args: argparse.Namespace, chain, index: int):
    from kersta.core.competition import Competition
    from kersta.core.constants import MIN_SCORING_BLOCKS, MIN_UPLOAD_BLOCKS

    if args.upload_blocks < MIN_UPLOAD_BLOCKS or args.scoring_blocks < MIN_SCORING_BLOCKS:
        raise SystemExit(f"the upload window needs at least {MIN_UPLOAD_BLOCKS} blocks and the "
                         f"scoring window at least {MIN_SCORING_BLOCKS}")
    deadline = args.deadline_block if args.deadline_block else chain.block() + args.deadline_in
    reveal = deadline + args.upload_blocks
    return Competition(index, deadline, reveal, reveal + args.scoring_blocks)


def _index_of(folder: Path) -> int:
    name = folder.resolve().name
    prefix = "competition-"
    if not name.startswith(prefix) or not name[len(prefix):].isdigit():
        raise SystemExit(f"{folder} is not a competition-<n> folder")
    return int(name[len(prefix):])


def _build(args: argparse.Namespace) -> int:
    from kersta.owner.factory import FactoryConfig, build

    seed = os.environ.get("KERSTA_TESTSET_SEED", "")
    if not seed:
        print("set KERSTA_TESTSET_SEED to a secret (e.g. `openssl rand -hex 32`)")
        return 1
    kwargs = {"items": args.items} if args.items else {}
    cfg = FactoryConfig(competition=args.competition, seed=seed, allow_reuse=args.allow_reuse,
                        **kwargs)
    lanes = [Lane.parse(v) for v in args.lane] if args.lane else None
    root, notes = build(Path(args.sources), Path(args.out), cfg, lanes)
    (root.parent / f"{root.name}.notes.json").write_text(json.dumps(notes, indent=2), "utf-8")
    print(f"built {root}")
    emit(notes)
    return 0


def _announce(args: argparse.Namespace) -> int:
    from kersta.owner.publish import announce

    chain, _, _ = open_chain(args, sign=True)
    print(f"committed {announce(chain, _competition(args, chain, args.competition))}")
    return 0


def _lock(args: argparse.Namespace) -> int:
    from kersta.owner.publish import lock, previous_on_chain, sign_competition

    folder = Path(args.folder)
    notes_path = folder.parent / f"{folder.resolve().name}.notes.json"
    notes = json.loads(notes_path.read_text("utf-8")) if notes_path.is_file() else {}
    chain, _, signer = open_chain(args, sign=True)
    assert signer is not None
    competition = _competition(args, chain, _index_of(folder))
    # The notes travel in the signed manifest: leak checks, generators, the
    # competition's blocks, the previous competition's blocks, and the hash of
    # the secret seed (never the seed itself).
    kept = {"lanes": notes.get("lanes", {}), "seed_sha256": notes.get("seed_sha256")}
    manifest = sign_competition(folder, competition, signer, kept,
                                previous_on_chain(chain, competition.index))
    print(f"signed competition {competition.index}: digest {manifest.digest}")
    print(f"committed {lock(chain, competition, manifest)}")
    return 0


def _publish(args: argparse.Namespace) -> int:
    from kersta.owner.publish import publish_to_folder

    print(f"published to {publish_to_folder(Path(args.folder), Path(args.store))}")
    return 0


def _offline_init(args: argparse.Namespace) -> int:
    from kersta.chain.offline import OfflineChain

    path = Path(args.chain)
    if path.exists():
        print(f"{path} exists")
        return 1
    chain = OfflineChain(path)
    chain.register(args.owner, validator=True, stake=100_000)
    chain.set_owner(args.owner)
    for hotkey in args.validator:
        chain.register(hotkey, validator=True, stake=10_000)
    for hotkey in args.miner:
        chain.register(hotkey)
    print(f"offline chain at {path}: owner {args.owner}, "
          f"{len(args.validator)} validators, {len(args.miner)} miners")
    return 0
