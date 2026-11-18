"""The ``kersta`` command."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from kersta import __version__
from kersta.cli import miner, owner, validator
from kersta.cli.common import add_chain_args, emit, open_chain, setup_logging
from kersta.core.competition import approx
from kersta.core.constants import MECHANISM_VERSION, NETUID
from kersta.core.lanes import Lane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kersta",
        description=f"Kersta deepfake detection subnet (netuid {NETUID}), "
                    f"release {__version__}, mechanism {MECHANISM_VERSION}",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    miner.register(sub)
    validator.register(sub)
    owner.register(sub)

    p = sub.add_parser("competition", help="show the competition the owner has announced")
    add_chain_args(p)
    p.set_defaults(func=_competition)

    p = sub.add_parser("damage", help="apply the damage tool to one file")
    p.add_argument("--lane", required=True, choices=[lane.value for lane in Lane])
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--chain", help="e.g. 'jpeg:quality=40;resize:scale=0.5' (default: random)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_damage)

    chain = sub.add_parser("chain", help="chain utilities")
    cmds = chain.add_subparsers(dest="chain_cmd", required=True)
    p = cmds.add_parser("doctor", help="check every chain read path against a live node")
    add_chain_args(p)
    p.set_defaults(func=_doctor)
    return parser


def _competition(args: argparse.Namespace) -> int:
    from kersta.chain.commitments import announced

    chain, _, _ = open_chain(args, sign=False)
    block = chain.block()
    found = announced(chain.commitments(), chain.owner_hotkey())
    print(f"block         {block}")
    if found is None:
        print("competition   none announced yet")
        return 0
    c = found.competition
    print(f"competition   {c.index} ({c.phase(block)} phase)")
    for name, at in (("deadline", c.deadline), ("reveal", c.reveal), ("end", c.end)):
        left = at - block
        when = f"in {left} blocks, {approx(left)}" if left > 0 else "passed"
        print(f"{name:<13} block {at}  ({when})")
    print(f"test set      {'locked ' + found.digest if found.locked else 'not locked yet'}")
    return 0


def _damage(args: argparse.Namespace) -> int:
    from kersta.media.damage import apply_to_file, parse_chain, sample_chain

    lane = Lane.parse(args.lane)
    chain = parse_chain(args.chain) if args.chain else sample_chain(lane, random.Random(args.seed))
    apply_to_file(lane, Path(args.src), Path(args.dst), chain)
    print("applied " + ";".join(op.tag() for op in chain))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    chain, _, _ = open_chain(args, sign=False)
    found: dict[str, object] = {}
    checks = {
        "block": lambda: chain.block(),
        "timestamp": lambda: chain.timestamp(chain.block()),
        "metagraph_neurons": lambda: len(chain.snapshot(refresh=True)),
        "commitments": lambda: len(chain.commitments()),
        "owner_hotkey": lambda: chain.owner_hotkey(),
        "last_step_block": lambda: chain.last_step_block(),
    }
    for name, call in checks.items():
        try:
            found[name] = call()
        except Exception as exc:  # noqa: BLE001 - report every failure
            found[name] = f"FAILED: {exc}"
    found["paths"] = getattr(chain, "paths", {})
    emit(found)
    return 0 if not any(str(v).startswith("FAILED") for v in found.values()) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
