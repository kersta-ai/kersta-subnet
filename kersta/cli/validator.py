"""kersta validator ...: run, score-competition, weights, leaderboard."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from kersta.cli.common import add_chain_args, chain_config, emit, open_chain


def register(sub: argparse._SubParsersAction) -> None:
    validator = sub.add_parser("validator", help="run a validator")
    cmds = validator.add_subparsers(dest="validator_cmd", required=True)

    for name, helptext in (("run", "the long running loop"),
                           ("score-competition", "score one competition now and print the result")):
        p = cmds.add_parser(name, help=helptext)
        if name == "score-competition":
            p.add_argument("competition", type=int)
        else:
            p.add_argument("--once", action="store_true", help="one pass, then exit")
        p.add_argument("--home", help="state and cache folder (default ~/.kersta/validator)")
        p.add_argument("--testset-url", help="test set store (https URL or folder)")
        p.add_argument("--sandbox", choices=["docker", "local"])
        p.add_argument("--gpu", help="GPU index for the sandbox, or 'none'")
        p.add_argument("--dry-run", action="store_true", help="no weights, no report upload")
        p.add_argument("--item-limit", type=int, help="dry runs only: first N items per lane")
        p.add_argument("--packages", help="offline runs: local package folder instead of HF")
        add_chain_args(p)
        p.set_defaults(func=_run if name == "run" else _score)

    p = cmds.add_parser("weights", help="show the weights this validator has on chain")
    p.add_argument("--home")
    p.set_defaults(func=_weights)

    p = cmds.add_parser("leaderboard", help="print the leaderboard of a scored competition")
    p.add_argument("competition", type=int)
    p.add_argument("--home")
    p.set_defaults(func=_leaderboard)


def _config(args: argparse.Namespace):
    from kersta.validator.config import ValidatorConfig

    config = ValidatorConfig(chain=chain_config(args))
    overrides: dict[str, object] = {}
    if getattr(args, "home", None):
        overrides["home"] = Path(args.home).expanduser()
    for name in ("testset_url", "sandbox", "item_limit"):
        if getattr(args, name, None) is not None:
            overrides[name] = getattr(args, name)
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "gpu", None):
        overrides["gpu"] = None if args.gpu == "none" else args.gpu
    return dataclasses.replace(config, **overrides)


def _context(args: argparse.Namespace):
    from kersta.validator.loop import build_context

    config = _config(args)
    chain, wallet, signer = open_chain(args, sign=True)
    packages = Path(args.packages) if getattr(args, "packages", None) else None
    return build_context(config, chain, wallet=wallet, signer=signer, local_packages=packages)


def _run(args: argparse.Namespace) -> int:
    from kersta.validator.loop import run

    return run(_context(args), once=args.once)


def _score(args: argparse.Namespace) -> int:
    from kersta.validator.competition import score_competition
    from kersta.validator.loop import observe
    from kersta.validator.report import leaderboard

    ctx = _context(args)
    observe(ctx)
    competition = ctx.state.competition(args.competition)
    if competition is None:
        print(f"competition {args.competition} is not known here; it must be announced on chain")
        return 1
    result = score_competition(ctx, competition)
    emit({"weights": result.weights, "failures": result.failures,
          "leaderboard": leaderboard(result.report)})
    return 0


def _weights(args: argparse.Namespace) -> int:
    from kersta.validator.state import State

    emit(State(_config(args).state_dir).weights() or {"weights": None})
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    from kersta.validator.report import leaderboard
    from kersta.validator.state import State

    report = State(_config(args).state_dir).report(args.competition)
    if report is None:
        print(f"competition {args.competition} has not been scored here")
        return 1
    emit(leaderboard(report))
    return 0
