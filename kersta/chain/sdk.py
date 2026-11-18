"""The only module that imports bittensor.

Built against the Bittensor SDK 11 migration guide. Every call tries the
SDK 11 surface first and falls back to raw substrate calls, and then to the
SDK 9/10 names, because the exact v11 call signatures could not be verified
when this was written. ``kersta chain doctor`` exercises each read path
against a live node and prints which one answered; run it on testnet 456
before the first competition and after every SDK upgrade.

Reads that must be identical on every validator (commitments at the
deadline block) go through substrate storage queries pinned to a block
hash, which do not depend on the SDK version at all.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Final

from kersta.chain.client import ChainError, with_retry
from kersta.chain.commitments import OnChainCommitment
from kersta.chain.config import NETWORKS, ChainConfig
from kersta.chain.metagraph import MetagraphSnapshot, snapshot_from
from kersta.chain.weights import WeightVector, version_key
from kersta.core.constants import METAGRAPH_TTL_SECONDS

log = logging.getLogger("kersta.chain.sdk")
STALE_SECONDS: Final[float] = 120.0
# The storage item really is spelled this way in subtensor.
LAST_STEP_STORAGE: Final[str] = "LastMechansimStepBlock"


def _bittensor() -> Any:
    try:
        import bittensor
    except ImportError as exc:
        raise ChainError("bittensor is not installed: pip install 'kersta[chain]'") from exc
    return bittensor


def _value(found: Any) -> Any:
    return getattr(found, "value", found)


def _bytes_of(raw: Any) -> bytes:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        text = raw[2:] if raw.startswith("0x") else raw
        try:
            return bytes.fromhex(text)
        except ValueError:
            return raw.encode("utf-8")
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        return bytes(int(b) for b in raw)
    return str(raw).encode("utf-8")


def decode_commitment_info(info: Any) -> str:
    """Pull the text out of a Commitments pallet ``info`` value.

    The pallet stores ``{"fields": [{"Raw<n>": <bytes>}]}``; depending on the
    decoder the bytes come back as hex, a list of ints, or nested once more.
    """
    info = _value(info)
    fields = info.get("fields") if isinstance(info, dict) else info
    if not fields:
        return ""
    first = fields[0]
    while isinstance(first, (list, tuple)) and first and isinstance(first[0], (dict, list, tuple)):
        first = first[0]
    if isinstance(first, dict):
        for key, raw in first.items():
            if str(key).startswith("Raw"):
                return _bytes_of(raw).decode("utf-8", errors="replace")
    return ""


class SubtensorChain:
    def __init__(self, config: ChainConfig, wallet: Any | None = None) -> None:
        self._config = config
        self._wallet = wallet
        self._client: Any | None = None
        self._archive: Any | None = None
        self._lock = threading.RLock()
        self._snapshot: MetagraphSnapshot | None = None
        self._snapshot_at = 0.0
        self._last_block = (0, 0.0)
        self.paths: dict[str, str] = {}  # which surface answered each call, for `doctor`

    # ------------------------------------------------------------ connection

    @property
    def netuid(self) -> int:
        return self._config.netuid

    @property
    def wallet(self) -> Any:
        if self._wallet is None:
            raise ChainError("this chain client has no wallet; it cannot sign")
        return self._wallet

    def _connect(self, endpoint: str) -> Any:
        bt = _bittensor()
        factory = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor", None)
        if factory is None:
            raise ChainError("this bittensor build exposes no Subtensor client")
        return with_retry("connect", lambda: factory(network=endpoint))

    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._connect(self._config.resolved_endpoint)
            return self._client

    def _archive_client(self) -> Any:
        with self._lock:
            if self._archive is None:
                endpoint = self._config.archive_endpoint or NETWORKS["archive"]
                self._archive = self._connect(endpoint)
            return self._archive

    def _substrate(self, client: Any | None = None) -> Any | None:
        client = client or self.client
        for name in ("substrate", "_substrate"):
            found = getattr(client, name, None)
            if found is not None:
                return found
        return None

    def _reconnect(self) -> None:
        with self._lock:
            old, self._client = self._client, None
        if old is not None:
            closer = getattr(old, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001
                    log.debug("close failed: %s", exc)

    def close(self) -> None:
        self._reconnect()
        archive, self._archive = self._archive, None
        if archive is not None and callable(getattr(archive, "close", None)):
            archive.close()

    # ------------------------------------------------------------ reads

    def _first(self, label: str, attempts: list[tuple[str, Callable[[], Any]]]) -> Any:
        errors = []
        for name, call in attempts:
            try:
                result = call()
            except (AttributeError, TypeError, NotImplementedError) as exc:
                errors.append(f"{name}: {exc}")
                continue
            self.paths[label] = name
            return result
        raise ChainError(f"no {label} reader worked: {'; '.join(errors)}")

    def block(self) -> int:
        def property_block() -> int:
            value = self.client.block
            if callable(value):
                raise TypeError("block is a method here")
            return int(value)

        def method_block() -> int:
            return int(self.client.get_current_block())

        block = int(with_retry("block", lambda: self._first(
            "block", [("v11 Subtensor.block", property_block),
                      ("v10 get_current_block", method_block)])))
        last, seen_at = self._last_block
        now = time.monotonic()
        if block != last:
            self._last_block = (block, now)
        elif now - seen_at > STALE_SECONDS:
            # The node stopped advancing; a fresh connection usually fixes it.
            log.warning("block %d has not moved for %.0f s; reconnecting", block, now - seen_at)
            self._reconnect()
            self._last_block = (block, now)
        return block

    def block_hash(self, block: int, *, archive: bool = False) -> str:
        client = self._archive_client() if archive else self.client
        substrate = self._substrate(client)

        def via_substrate() -> str:
            if substrate is None:
                raise AttributeError("no substrate interface")
            return str(substrate.get_block_hash(block))

        def via_info() -> str:
            info = client.block_info(block)
            return str(getattr(info, "hash", None) or info.block_hash)

        def via_legacy() -> str:
            return str(client.get_block_hash(block))

        return str(with_retry("block hash", lambda: self._first(
            "block_hash", [("substrate.get_block_hash", via_substrate),
                           ("v11 block_info", via_info), ("v10 get_block_hash", via_legacy)])))

    def timestamp(self, block: int) -> int:
        def via_client() -> int:
            value = self.client.timestamp(block)
            return int(value.timestamp()) if hasattr(value, "timestamp") else int(value)

        def via_storage() -> int:
            substrate = self._substrate()
            if substrate is None:
                raise AttributeError("no substrate interface")
            found = substrate.query("Timestamp", "Now", block_hash=self.block_hash(block))
            return int(_value(found)) // 1000

        return int(with_retry("timestamp", lambda: self._first(
            "timestamp", [("v11 Subtensor.timestamp", via_client),
                          ("Timestamp.Now storage", via_storage)])))

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot:
        with self._lock:
            fresh = time.monotonic() - self._snapshot_at < METAGRAPH_TTL_SECONDS
            if self._snapshot is not None and fresh and not refresh:
                return self._snapshot

        def v11() -> Any:
            return self.client.subnets.metagraph(netuid=self.netuid)

        def v10() -> Any:
            return self.client.metagraph(netuid=self.netuid)

        raw = with_retry("metagraph", lambda: self._first(
            "metagraph", [("v11 subnets.metagraph", v11), ("v10 metagraph", v10)]))
        snapshot = snapshot_from(raw, netuid=self.netuid, block=self.block())
        with self._lock:
            self._snapshot, self._snapshot_at = snapshot, time.monotonic()
        return snapshot

    def commitments(self, at_block: int | None = None) -> list[OnChainCommitment]:
        """Commitments as they stood after ``at_block``, with the block each was written.

        Reads storage pinned to a block hash. When the node has pruned that
        state, it retries on the archive endpoint.
        """
        def read(archive: bool) -> list[OnChainCommitment]:
            client = self._archive_client() if archive else self.client
            substrate = self._substrate(client)
            if substrate is None:
                raise ChainError("this bittensor build exposes no substrate interface")
            block_hash = None if at_block is None else self.block_hash(at_block, archive=archive)
            rows = substrate.query_map(
                "Commitments", "CommitmentOf", [self.netuid], block_hash=block_hash
            )
            found = []
            for key, value in rows:
                record = _value(value)
                if not isinstance(record, dict):
                    continue
                text = decode_commitment_info(record.get("info"))
                if text:
                    hotkey = str(_value(key))
                    found.append(OnChainCommitment(hotkey, int(record.get("block", 0)), text))
            self.paths["commitments"] = "archive" if archive else "node"
            return found

        try:
            return with_retry("commitments", lambda: read(False), attempts=2)
        except ChainError as exc:
            if at_block is None:
                raise
            log.warning("state at block %s unavailable on the node, using archive: %s",
                        at_block, exc)
            return with_retry("commitments (archive)", lambda: read(True))

    def owner_hotkey(self) -> str:
        def via_storage() -> str:
            substrate = self._substrate()
            if substrate is None:
                raise AttributeError("no substrate interface")
            found = substrate.query("SubtensorModule", "SubnetOwnerHotkey", [self.netuid])
            return str(_value(found) or "")

        def via_v10() -> str:
            return str(self.client.get_subnet_owner_hotkey(self.netuid) or "")

        return str(with_retry("owner hotkey", lambda: self._first(
            "owner_hotkey", [("SubnetOwnerHotkey storage", via_storage),
                             ("v10 get_subnet_owner_hotkey", via_v10)])))

    def last_step_block(self) -> int | None:
        substrate = self._substrate()
        if substrate is None:
            return None
        try:
            found = substrate.query("SubtensorModule", LAST_STEP_STORAGE, [self.netuid])
            return int(_value(found) or 0) or None
        except Exception as exc:  # noqa: BLE001
            log.debug("last step block unreadable: %s", exc)
            return None

    # ------------------------------------------------------------ writes

    def commit(self, payload: str) -> None:
        data = payload.encode("utf-8")
        info = {"fields": [[{f"Raw{len(data)}": data}]]}
        bt = _bittensor()
        wallet = self.wallet

        def v11() -> Any:
            call = bt.calls.Commitments.set_commitment(netuid=self.netuid, info=info)
            return self.client.submit_call(call, wallet, signer="hotkey")

        def raw() -> Any:
            substrate = self._substrate()
            if substrate is None:
                raise AttributeError("no substrate interface")
            call = substrate.compose_call(
                call_module="Commitments",
                call_function="set_commitment",
                call_params={"netuid": self.netuid, "info": info},
            )
            extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.hotkey)
            receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            if not getattr(receipt, "is_success", True):
                raise ChainError(f"commitment refused: {getattr(receipt, 'error_message', '')}")
            return receipt

        def v10() -> Any:
            writer = getattr(self.client, "set_commitment", None) or self.client.commit
            return writer(wallet, self.netuid, payload)

        result = with_retry("commit", lambda: self._first(
            "commit", [("v11 submit_call", v11), ("raw set_commitment", raw),
                       ("v10 set_commitment", v10)]), attempts=2)
        ok, message = _outcome(result)
        if not ok:
            raise ChainError(f"commitment not accepted: {message}")

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]:
        if vector.is_empty:
            return False, "refusing to submit an empty weight vector"
        bt = _bittensor()
        weights = vector.as_dict()
        key = version_key()

        def v11() -> Any:
            intent_type = bt.SetWeights
            try:
                intent = intent_type(netuid=self.netuid, weights=weights, mechid=0, version_key=key)
            except TypeError:
                log.warning("SetWeights takes no version_key here; submitting without it")
                intent = intent_type(netuid=self.netuid, weights=weights, mechid=0)
            return self.client.execute(intent, self.wallet)

        def v10() -> Any:
            return self.client.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=list(vector.uids),
                weights=list(vector.values),
                version_key=key,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

        try:
            # Not retried here: the keeper resends a refused vector minutes later.
            result = self._first("set_weights", [("v11 SetWeights", v11), ("v10 set_weights", v10)])
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return _outcome(result)


def _outcome(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    for attr in ("success", "is_success", "ok"):
        if hasattr(result, attr):
            message = getattr(result, "message", "") or getattr(result, "error_message", "")
            return bool(getattr(result, attr)), str(message or "")
    return (True, "") if result is None else (bool(result), "")
