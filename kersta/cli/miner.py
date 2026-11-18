"""kersta miner ...: init, check, hash, score, submit, reveal, status."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kersta.cli.common import add_chain_args, emit, open_chain
from kersta.core.lanes import Lane
from kersta.core.package import PackageError, scan_submission


def register(sub: argparse._SubParsersAction) -> None:
    miner = sub.add_parser("miner", help="build, check, score and submit a detector")
    cmds = miner.add_subparsers(dest="miner_cmd", required=True)

    p = cmds.add_parser("init", help="copy a starter package for a lane")
    p.add_argument("--lane", required=True, choices=[lane.value for lane in Lane])
    p.add_argument("--out", default="submission", help="submission folder")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_init)

    p = cmds.add_parser("check", help="validate the package and smoke run it")
    p.add_argument("path", nargs="?", default="submission")
    p.add_argument("--no-run", action="store_true", help="static checks only")
    p.set_defaults(func=_check)

    p = cmds.add_parser("hash", help="print the submission digest")
    p.add_argument("path", nargs="?", default="submission")
    p.set_defaults(func=_hash)

    p = cmds.add_parser("score", help="score locally on a released test set")
    p.add_argument("path", nargs="?", default="submission")
    p.add_argument("--testset", required=True, help="a released competition-<n> folder")
    p.add_argument("--lane", choices=[lane.value for lane in Lane])
    p.add_argument("--limit", type=int, help="first N items only (quick runs)")
    p.set_defaults(func=_score)

    p = cmds.add_parser("submit", help="upload to Hugging Face (private) and commit on chain")
    p.add_argument("path", nargs="?", default="submission")
    p.add_argument("--repo", required=True, help="Hugging Face repo id, user/name")
    p.add_argument("--public", action="store_true", help="upload public now (copy risk)")
    p.add_argument("--no-commit", action="store_true", help="upload only")
    add_chain_args(p)
    p.set_defaults(func=_submit)

    p = cmds.add_parser("reveal", help="make the repo public (after the Monday deadline)")
    p.add_argument("--repo", required=True)
    p.set_defaults(func=_reveal)

    p = cmds.add_parser("status", help="show your commitment and the competition it enters")
    add_chain_args(p)
    p.set_defaults(func=_status)


def _init(args: argparse.Namespace) -> int:
    from kersta.miner.package import init

    target = init(args.lane, Path(args.out), force=args.force)
    print(f"starter {args.lane} package copied to {target}")
    print(f"next: train weights with `python -m kersta.starter.{args.lane}.train --help`, "
          f"save them as {target / 'model.safetensors'}, then `kersta miner check {args.out}`")
    return 0


def _check(args: argparse.Namespace) -> int:
    from kersta.miner.package import check

    try:
        report = check(Path(args.path), smoke=not args.no_run)
    except PackageError as exc:
        print(f"refused: {exc}")
        return 1
    emit(report.to_dict())
    return 0 if report.ok else 1


def _hash(args: argparse.Namespace) -> int:
    submission = scan_submission(Path(args.path))
    print(submission.digest)
    for lane, package in submission.lanes.items():
        print(f"  {lane.value:<6} {package.digest}  {package.size:,} bytes")
    return 0


def _score(args: argparse.Namespace) -> int:
    from kersta.miner.score import render, score_local

    lanes = [Lane.parse(args.lane)] if args.lane else None
    scores = score_local(Path(args.path), Path(args.testset), lanes=lanes, limit=args.limit)
    print(render(scores))
    return 0


def _submit(args: argparse.Namespace) -> int:
    from kersta.miner.submit import commit, upload

    token = os.environ.get("HF_TOKEN")
    revision, digest = upload(Path(args.path), args.repo, token=token, private=not args.public)
    print(f"uploaded {args.repo}@{revision}\ndigest   {digest}")
    if args.no_commit:
        return 0
    from kersta.chain.commitments import announced
    from kersta.miner.submit import entry_line

    chain, _, _ = open_chain(args, sign=True)
    payload = commit(chain, args.repo, revision, digest)
    block = chain.block()
    lock = announced(chain.commitments(), chain.owner_hotkey())
    print(f"committed {payload}")
    print(entry_line(lock.competition if lock else None, block, block))
    if not args.public:
        print("remember: `kersta miner reveal --repo ...` after the deadline block, "
              "before the reveal block")
    return 0


def _reveal(args: argparse.Namespace) -> int:
    from kersta.miner.submit import reveal

    reveal(args.repo, token=os.environ.get("HF_TOKEN"))
    print(f"{args.repo} is public")
    return 0


def _status(args: argparse.Namespace) -> int:
    from kersta.miner.submit import status

    chain, wallet, signer = open_chain(args, sign=True)
    hotkey = signer.address if signer else ""
    for line in status(chain, hotkey).lines():
        print(line)
    return 0
