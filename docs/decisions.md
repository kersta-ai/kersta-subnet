# Decisions

The blueprint fixes the mechanism: lanes, test set mix, scoring formulas and payout. This page records the choices the blueprint leaves open, how each was settled in mechanism 1.0.0, the decisions still open, and what remains to be verified before the first competition on testnet 456.

## Settled in mechanism 1.0.0

| Topic | Decision | Reason |
|---|---|---|
| Item failures | More than 1% of items failed scores 0. Below that, a failed item counts as `p_fake` 0.5 with no location. | A crash must cost something without one bad file being catastrophic |
| Empty slice | Left out of D, with the other weights renormalised | Only arises in small development sets |
| Smoothing | 70% of this competition's rank shares plus 30% of the previous competition's own shares. If this one pays nobody, the previous one's 30% is paid and 70% burns. | A two competition blend, not an unbounded average |
| Burns | Empty lanes, unpaid smoothing remainders and hotkeys that left all go to the owner hotkey | Emissions never reward work that did not clear the bar |
| Competition timing | The owner announces each competition on chain as three blocks (deadline, reveal, end); commitments are read as of the block before the deadline | No dates and no fixed cycle; every validator derives the same cut off from the chain itself |
| Commitment digest | The first 32 hex characters (128 bits) of the sha256 | Repository id, revision and full digest do not fit the 128 byte commitment |
| Submission layout | One repository per miner, one folder per lane | One commitment slot per hotkey; any subset of lanes can be entered |
| Timing | One item per call, timed by the validator outside the container, after one warm up call; p95 | Cannot be influenced by the model; matches call and meeting latency |
| Early stop | Only when the zero is already certain: failures above 1%, or items over the limit above 5%, of the whole set | Saves GPU time without changing any result |
| Default inputs | `uint8` pixels; the model normalises | A quarter of the bytes through the pipe; normalisation is faster on the GPU |
| Plain classifier packages | A torch module returning class logits runs through an adapter, with no location | Existing detectors can be submitted without rewriting them |
| Test set storage | A static store at `KERSTA_TESTSET_URL/competition-<n>/`, with an optional bearer token | The simple choice the blueprint names; any bucket or CDN works |
| Audio storage | 16 kHz PCM WAV after codec damage | Identical decoding everywhere, and no sample rate difference between real and fake |
| Version key | `MECHANISM_VERSION` (1.0.0 is 10000), separate from the release version | A rule change can be enforced on chain; a bug fix needs no coordinated bump |

## Open decisions

1. **MCC rescaling.** The blueprint specifies (MCC + 1)/2, under which a coin flip earns 0.5 on that factor. `max(0, MCC)` would give random guessing nothing. It is a one line change in `kersta/scoring/metrics.py` and a mechanism bump. It is kept as specified.
2. **Video compute budget.** 4,000 clips at up to 2 s each is up to 2.2 GPU hours per model, which does not fit the default scoring window of 3,600 blocks on one GPU with many miners. The options:
   - 1,000 video items per competition;
   - a lower video speed limit;
   - two GPUs for video validators;
   - a per validator subsample seeded from the chain.

   Decide before Phase 3 gives video full weight. Until then the video lane burns its third whenever no video test set is released.
3. **Phase 1 lane split.** While only image and audio run, video's third burns. Setting `LANE_SHARES` to one half each for Phase 1 pays miners more, and is a mechanism change.
4. **Reference GPU** for the speed limits, and whether validators on faster cards should apply a correction measured against the starter detector.
5. **Resubmission fee.** Version 1 has none. If commitment spam appears, add one, for example an alpha burn for each resubmission after the first.
6. **Hosting** on rented GPUs or on a decentralised GPU network. This does not affect the subnet code.
7. **License** for this repository.

## To verify before competition 0

The chain calls were written against the Bittensor SDK 11 migration guide, without an SDK installed, and have not run against a live chain:

- [ ] `kersta chain doctor` on testnet 456: block, timestamps, metagraph, commitments, owner hotkey, last step block.
- [ ] Writing a commitment with SDK 11 (`bt.calls.Commitments.set_commitment` and `submit_call`, falling back to a raw `compose_call`).
- [ ] Setting weights with `bt.SetWeights(netuid, weights={uid: w}, mechid=0, version_key=...)` and `execute`; confirm the version key argument and the return value.
- [ ] Wallet signing and verification through `bittensor.wallet`.
- [ ] Training the starter detectors on a GPU, then `kersta miner check` on the result.
- [ ] Building the sandbox image and running one model through `DockerLauncher`.
- [ ] The ffmpeg paths: audio codec damage, video decoding and video damage.

All chain calls live in `kersta/chain/sdk.py`. Once verified, remove the fallbacks SDK 11 does not need.
