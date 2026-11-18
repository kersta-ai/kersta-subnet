# Mechanism

Miners ship a deepfake detector for one or more lanes (image, video, audio) as one Hugging Face repository and commit its digest on chain. The owner decides when each competition runs and announces it from the owner hotkey with three blocks: a deadline, a reveal and an end. Each competition:

- Validators read the commitments as they stood just before the deadline block, and verify the owner's test set against the digest it locked on chain.
- After the reveal block, they run every package in a locked down Docker container on their own GPUs:
  - no network;
  - read only package and root filesystem;
  - every capability dropped;
  - one item at a time, timed from outside.
- They score each lane on its clean, damaged and unseen generator slices. Calibration is half of every slice score, and false alarms on real media halve the score.
- The top twenty per lane are paid with a 0.8 decay, blended 70/30 with the previous competition, one third per lane. Whatever nobody earned is burned to the owner hotkey.
- Validators set the weights themselves at the end block and keep them fresh every chain epoch until the next competition ends.

Every phase is a block height on chain; nothing depends on a date or a clock. `kersta competition` prints the competition the owner has announced and its blocks.

The design, with every formula, is in the [README](../README.md), sections 1 to 12. The documents:

- [package_spec.md](package_spec.md): the detector package, version 1
- [miner_setup.md](miner_setup.md): mining, and the training playbook
- [miner.md](miner.md): building, checking and submitting a detector
- [validator_setup.md](validator_setup.md): running a validator
- [test_sets.md](test_sets.md): announcing a competition, and building, locking and releasing its test set
- [decisions.md](decisions.md): choices beyond the blueprint, open decisions, and what is still to be verified

Code map:

| Path | Holds |
|---|---|
| `kersta/core` | lanes, package, outputs, test set format, competition blocks, constants |
| `kersta/chain` | commitments, metagraph, weights, `sdk.py` |
| `kersta/media` | preprocessing and the damage tool |
| `kersta/sandbox` | policy, worker, runner, protocol |
| `kersta/scoring` | metrics, lane score, copy check, payout |
| `kersta/validator` | the competition pipeline |
| `kersta/miner`, `kersta/owner` | the tools behind `kersta miner` and `kersta owner` |
