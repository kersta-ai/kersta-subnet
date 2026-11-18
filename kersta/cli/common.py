from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from kersta.chain.client import Chain
from kersta.chain.config import ChainConfig
from kersta.chain.offline import OfflineChain
from kersta.chain.wallet import DevSigner, Signer, WalletSigner, load_wallet


def add_chain_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("chain")
    group.add_argument("--network", help="test (default), finney, local, or a ws(s):// URL")
    group.add_argument("--netuid", type=int, help="subnet id (default 456)")
    group.add_argument("--archive-endpoint", help="node with old state, for deadline snapshots")
    group.add_argument("--wallet.name", dest="wallet_name", help="coldkey name")
    group.add_argument("--wallet.hotkey", dest="wallet_hotkey", help="hotkey name")
    group.add_argument("--wallet.path", dest="wallet_path", help="wallets folder")
    group.add_argument("--offline-chain", metavar="FILE",
                       help="use a local JSON chain instead of a real network (development)")
    group.add_argument("--dev-hotkey", help="with --offline-chain: the hotkey to act as")


def chain_config(args: argparse.Namespace) -> ChainConfig:
    network = getattr(args, "network", None)
    endpoint = None
    if network and network.startswith(("ws://", "wss://")):
        endpoint, network = network, None
    if getattr(args, "offline_chain", None):
        network = "offline"
    return ChainConfig.from_env(
        network=network, endpoint=endpoint, netuid=getattr(args, "netuid", None),
        archive_endpoint=getattr(args, "archive_endpoint", None),
        wallet_name=getattr(args, "wallet_name", None),
        wallet_hotkey=getattr(args, "wallet_hotkey", None),
        wallet_path=getattr(args, "wallet_path", None),
    )


def open_chain(args: argparse.Namespace, *, sign: bool) -> tuple[Chain, Any, Signer | None]:
    """(chain, wallet or None, signer or None)."""
    config = chain_config(args)
    if config.offline:
        hotkey = getattr(args, "dev_hotkey", None) or f"dev-{config.wallet_hotkey}"
        chain = OfflineChain(args.offline_chain, netuid=config.netuid, hotkey=hotkey)
        return chain, None, DevSigner(hotkey)
    from kersta.chain.sdk import SubtensorChain

    wallet = load_wallet(config.wallet_name, config.wallet_hotkey, config.wallet_path) if sign else None
    return SubtensorChain(config, wallet), wallet, WalletSigner(wallet) if wallet else None


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
