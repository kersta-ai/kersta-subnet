"""Owner tools: build the weekly test set, sign it, lock it on chain, publish it.

In version 1 the owner builds the test set, as the spec says. Validators do
not trust the owner blindly: the set must be signed by the owner hotkey,
its digest must be on chain before the deadline, and every file is hashed.
"""
