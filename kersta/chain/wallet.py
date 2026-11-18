"""Wallets and hotkey signatures.

Validators sign their competition reports and the owner signs test set manifests
with sr25519 hotkeys. ``DevSigner`` exists only for the offline chain and
tests; its "signatures" are hashes anyone can forge, and the validator
refuses them on a real network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from kersta.chain.client import ChainError

DEV_PREFIX = "dev:"


class Signer(Protocol):
    @property
    def address(self) -> str: ...

    def sign(self, message: bytes) -> str: ...


def load_wallet(name: str, hotkey: str, path: str = "") -> Any:
    try:
        from bittensor.wallet import Wallet  # SDK 11
    except ImportError:
        try:
            from bittensor_wallet import Wallet  # SDK 9 / 10
        except ImportError as exc:
            raise ChainError("bittensor is not installed: pip install 'kersta[chain]'") from exc
    kwargs: dict[str, Any] = {"name": name, "hotkey": hotkey}
    if path:
        kwargs["path"] = str(Path(path).expanduser())
    return Wallet(**kwargs)


class WalletSigner:
    def __init__(self, wallet: Any) -> None:
        self._wallet = wallet

    @property
    def address(self) -> str:
        return str(self._wallet.hotkey.ss58_address)

    def sign(self, message: bytes) -> str:
        signature = self._wallet.hotkey.sign(message)
        return "0x" + (signature.hex() if isinstance(signature, (bytes, bytearray)) else str(signature))


class DevSigner:
    def __init__(self, address: str) -> None:
        self._address = address

    @property
    def address(self) -> str:
        return self._address

    def sign(self, message: bytes) -> str:
        return DEV_PREFIX + hashlib.sha256(self._address.encode() + message).hexdigest()


def verify(address: str, message: bytes, signature: str, *, allow_dev: bool = False) -> bool:
    if signature.startswith(DEV_PREFIX):
        expected = DEV_PREFIX + hashlib.sha256(address.encode() + message).hexdigest()
        return allow_dev and signature == expected
    try:
        try:
            from bittensor.wallet import Keypair  # SDK 11
        except ImportError:
            from bittensor_wallet import Keypair  # SDK 9 / 10
    except ImportError as exc:
        raise ChainError("bittensor is not installed: pip install 'kersta[chain]'") from exc
    body = signature[2:] if signature.startswith("0x") else signature
    try:
        return bool(Keypair(ss58_address=address).verify(message, bytes.fromhex(body)))
    except (ValueError, TypeError):
        return False
