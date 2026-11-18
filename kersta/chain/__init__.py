"""Everything that touches the chain.

Only ``sdk.py`` imports bittensor. The rest of Kersta talks to the
``Chain`` protocol in ``client.py``, which ``SubtensorChain`` (real chain)
and ``OfflineChain`` (a JSON file, for tests and local competitions) implement.
"""
