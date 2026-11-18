# Running a competition

The owner decides when each competition runs, announces it on chain, and builds its test set. Validators verify the test set and never take it on trust. A set is accepted only after three checks:
- it is signed by the owner hotkey;
- its digest was locked on chain before the competition's deadline block;
- every file matches its listed sha256.

The owner never scores.

Code: `kersta/owner/factory.py`, `kersta/owner/publish.py`, `kersta/core/competition.py`.

## Requirements

| | |
|---|---|
| Sources | licensed real media captured fresh for each competition; consented voice and video; fakes per generator, including partial edits with an exact answer |
| Software | Python 3.10 or newer, ffmpeg (audio codecs, video), the owner hotkey |
| Store | a static host (bucket or CDN) serving `competition-<n>/` under `KERSTA_TESTSET_URL`, optionally behind a bearer token |
| Secret | `KERSTA_TESTSET_SEED`, never on a validator and never in git |

## The three blocks

A competition is three block heights, chosen by the owner:

| Block | Meaning |
|---|---|
| deadline | miner commitments written in an earlier block enter the competition; the test set must be locked before it |
| reveal | test files are released to validators; the blocks from the deadline to the reveal are the upload window, in which miners make their repositories public |
| end | scoring closes and validators set weights |

The tools default to 1,800 blocks from the deadline to the reveal and 3,600 from the reveal to the end. Either gap can be set per competition.

## 1. Announce

```bash
kersta owner announce --competition <n> --deadline-in <blocks> --wallet.name <coldkey> --wallet.hotkey <owner hotkey>
```

This commits `kc1|<n>|<deadline>|<reveal>|<end>|-` from the owner hotkey, so miners can see the deadline with `kersta competition` before the test set exists. `--deadline-block` takes an absolute block instead of `--deadline-in`. Announcing is optional; the lock below announces as well.

## 2. Sources

```text
sources/
  generators.json                     {"image": ["gen_a", "gen_b"], "audio": [...]}: used in earlier competitions
  image/real/**/*.png|jpg|webp
  image/fake/<generator>/**/*         fully generated
  image/partial/<generator>/x.png     partial edits (face swap, inpainting) ...
  image/partial/<generator>/x.mask.png   ... white where edited, same size
  audio/real/**/*.wav|flac|mp3
  audio/fake/<generator>/**/*
  audio/partial/<generator>/x.wav + x.segments.json   [[start_s, end_s], ...]
  video/... (same pattern)
```

Any generator not in `generators.json` counts as unseen in this competition. After a build, the factory writes `generators.after-competition-<n>.json`, which becomes the next competition's `generators.json`.

## 3. Build

```bash
KERSTA_TESTSET_SEED=<secret> kersta owner build-testset --sources sources/ --out built/ --competition <n>
```

The factory:

1. **Draws the mix.** It draws 4,000 items per lane: half real, half fake. 20% of the fakes come from unseen generators and about 40% are partial edits. Generators are drawn evenly, not by file count.
2. **Damages.** It damages 70% of the real items and 70% of the fake items, each by a chain drawn from the same distribution without looking at the label:
   - image: compression, resizing, cropping, screenshots, recapture, social uploads;
   - audio: phone line, codecs, re-recording, noise, resampling;
   - video: re-encoding, scaling, cropping, frame rate.
3. **Normalises.** It re-encodes every file to one format per lane (PNG, 16 kHz WAV, H.264 MP4), strips metadata, and renames each file to a random id.
4. **Removes size and length clues.** It caps images to one long edge, and draws clip lengths the same way for real and fake.
5. **Fills the unseen slice.** It adds a random set of real items equal in number to the unseen fakes.
6. **Writes the answer key** (`<lane>/answers.jsonl`) and the notes. The notes give per label mean clip length and image size, so a leak shows before release, and the sha256 of the seed. The seed itself never leaves the owner.

Source hygiene still matters. If the real photos come from one camera and the fakes from one export format, traces of that remain in the pixels after re-encoding. Match sources: the same subjects, the same channels, matched real and cloned voice pairs.

## 4. Lock

```bash
kersta owner lock built/competition-<n> --deadline-block <block> --wallet.name <coldkey> --wallet.hotkey <owner hotkey>
```

`lock` hashes every file into `manifest.json` and signs it with the owner hotkey. The signed manifest also carries:
- the competition's three blocks;
- the previous competition's blocks, read from the chain, so a validator that joins late can rebuild its smoothing;
- the build notes.

It then commits `kc1|<n>|<deadline>|<reveal>|<end>|<digest32>`. It refuses once the deadline block has passed, because a lock after the deadline would not count.

The owner hotkey has one commitment slot. Validators read each competition's lock from the snapshot at its own deadline, so announcing the next competition afterwards does not affect the one being scored.

## 5. Release

| Block | Step |
|---|---|
| reveal | Sync `built/competition-<n>/` to the store as `competition-<n>/`, for example `aws s3 sync built/competition-<n> s3://<bucket>/competition-<n>` |
| after the end | Publish the same folder publicly as training data |

`kersta owner publish built/competition-<n> --store <folder>` copies a competition into a folder store, for offline runs or before syncing to a bucket.

## Local competitions without TAO

```bash
kersta owner offline-init --chain dev/chain.json --owner owner --validator v1 --miner alice --miner bob
```

Then add `--offline-chain dev/chain.json --dev-hotkey <who>` to any command. `tests/test_competition_flow.py` scripts two full competitions:
1. the owner announces and locks a competition;
2. three miners commit, one of them copying another;
3. it is scored at the reveal block and weights are set at the end block;
4. a second competition is smoothed with the first, and a validator that joins late reaches the same weights.

## Before launch

- [ ] Train and publish the starter weights for each lane, and record their scores as the baseline.
- [ ] Pin the sandbox image by digest in `KERSTA_SANDBOX_IMAGE` and publish it.
- [ ] Choose and publish the reference GPU, and measure the starters on it.
- [ ] Run `kersta chain doctor` on testnet 456 and set weights once from a test validator.
- [ ] Set the subnet's `WeightsVersionKey` to 10000 (mechanism 1.0.0).
