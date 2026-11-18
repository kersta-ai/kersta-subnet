<h1 align="center">Kersta</h1>

<p align="center"><b>Verified deepfake detection, trained by an open network and measured in every competition on hidden, damaged media.</b></p>

<p align="center">
  <a href="https://bittensor.com"><img src="https://img.shields.io/badge/Bittensor-subnet-6d7cff" alt="Bittensor subnet"></a>
  <img src="https://img.shields.io/badge/status-testnet%20456-f5b544" alt="Status: testnet 456">
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10%2B-34d399" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/license-to%20be%20decided-8b93a1" alt="License: to be decided">
</p>

<p align="center">
  <a href="docs/mechanism.md">Mechanism</a> ·
  <a href="docs/package_spec.md">Package</a> ·
  <a href="docs/miner_setup.md">Mining</a> ·
  <a href="docs/validator_setup.md">Validating</a> ·
  <a href="#9-participating">Participate</a>
</p>

---

> **Abstract.** Generated images, cloned voices and face swapped video are now cheap to produce and good enough to pass a human check, and they are used against banks, contact centres, recruiters and newsrooms. Detectors exist, but the evidence needed to trust one does not. Public benchmarks are clean and leak into training data, and open detectors lose close to half their AUC on media circulating in the wild. Kersta is a Bittensor subnet that treats **robustness under real conditions as the measured quantity**. The owner runs competitions, each defined by blocks on chain, and for each *lane* (image, video, audio) builds a fresh hidden test set for every competition. Half of it is real and half fake; 70% has been through the compression, resizing, screenshots, re-recording and phone codecs that real media goes through; and 20% of the fakes come from generators no earlier competition used. Its digest is locked on chain before miners' packages freeze. Miners submit a detector as weights and code on Hugging Face and commit its digest on chain; they host nothing. Every validator downloads every committed package, runs it in a sandbox with no network, one item at a time, and scores detection, calibration and the location of the fake region. The output is a detector whose published accuracy is a measurement on media it has never seen, served through one API.

## Contents

1. [The problem](#1-the-problem)
2. [Approach: measurement under real conditions](#2-approach-measurement-under-real-conditions)
3. [System model](#3-system-model)
4. [Competition protocol](#4-competition-protocol)
5. [Evaluation](#5-evaluation)
6. [Scoring](#6-scoring)
7. [Incentives](#7-incentives)
8. [Threat model](#8-threat-model)
9. [Participating](#9-participating)
10. [Lanes](#10-lanes)
11. [Repository layout](#11-repository-layout)
12. [Citation and license](#12-citation-and-license)

---

## 1. The problem

### 1.1 Synthetic media now reaches people who act on it

A fake only costs money when someone acts on it: a cloned voice calling a bank, a face swap passing an identity check, a synthetic candidate in a video interview, a fabricated photo in the news. The buyers of detection are the parties who act.

| Buyer | What they need checked | Product they use |
|---|---|---|
| Banks and fintechs | Selfie videos at account opening, voice calls to support | API, call plugin |
| Contact centres | Cloned voices and AI callers | Call plugin |
| Recruiters and HR | Synthetic candidates in video interviews | Meeting plugin |
| Identity verification providers | Face swaps and fake documents | API, private install |
| Newsrooms and investigators | Photos and videos before publication | Scan app |
| The public | Anything suspicious online | Browser extension, free scans |

### 1.2 Detection works in the lab and degrades in the wild

The literature is consistent on where detectors fail:

| Finding | Source |
|---|---|
| Open source detectors lose **45% (image), 48% (audio) and 50% (video) of their AUC** on deepfakes circulating in 2024, compared with earlier benchmarks | Chandra et al., 2025, [arXiv:2503.02857](https://arxiv.org/abs/2503.02857) |
| A detector trained on GAN images fails on newer families of generators; a detector built on general purpose features improved on the state of the art by **+15.07 mAP and +25.90% accuracy** on unseen diffusion and autoregressive models | Ojha et al., 2023, [arXiv:2302.10174](https://arxiv.org/abs/2302.10174) |
| With careful pre and post processing and **data augmentation**, a classifier trained on one generator generalises surprisingly well to unseen architectures | Wang et al., 2020, [arXiv:1912.11035](https://arxiv.org/abs/1912.11035) |
| Modern neural networks are **poorly calibrated**; temperature scaling is surprisingly effective at correcting it | Guo et al., 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| A benchmark of **40 deepfake techniques** argues that training on older forgeries makes it hard to guarantee generalisation to current ones | Yan et al., 2024, [arXiv:2406.13495](https://arxiv.org/abs/2406.13495) |
| Speech anti spoofing has moved to crowdsourced data from many speakers in **diverse acoustic conditions**, with adversarial attacks included for the first time | Wang et al., 2024, [arXiv:2408.08739](https://arxiv.org/abs/2408.08739) |

The ability to detect exists. What is missing is evidence of how a given detector behaves on the media a buyer will actually send it.

### 1.3 The measurement gap

A buyer choosing a detector today relies on numbers that fail in four ways:

- **Clean tests.** Benchmarks are mostly pristine files, while media in use has been recompressed, resized, screenshotted, re-recorded and sent over phone lines.
- **Contamination.** Public test sets are scraped into training corpora over time, so a high score cannot separate a detector that generalises from one that memorised the test.
- **Overconfidence.** A probability of 0.99 is only useful if it is right 99% of the time, and accuracy alone does not measure that.
- **No explanation.** A score with no indication of where the media is fake cannot be checked by the person who has to act on it.

**Research question.** *Can an open, adversarial market produce deepfake detectors whose accuracy, calibration and localisation on damaged, never seen media are measured well enough for a bank to act on, at a cost validators can afford to repeat in every competition?*

## 2. Approach: measurement under real conditions

Kersta measures detectors under the conditions they will be used in, on tests nobody has seen, and pays for the measurement rather than for claims. Seven design principles follow from that.

| | Principle | Mechanism |
|---|---|---|
| **P1** | Tests nobody has seen | 4,000 fresh items per lane in every competition. The digest is locked on chain before the deadline and the files are released after packages freeze. Each set is used once, then published as training data. |
| **P2** | Real conditions, not lab conditions | 70% of items are damaged, real and fake at the same rate, by chains drawn without looking at the label |
| **P3** | Generalisation, not memory | 20% of fakes come from generators no earlier competition used, scored as their own slice |
| **P4** | Confidence that means something | Calibration is half of every slice score; false alarms on real media above 2% halve the score |
| **P5** | Say where, not only whether | Location of partial edits (a pixel mask, or time segments) is 15% of the score |
| **P6** | The measured detector is the shipped detector | The digest committed on chain is the package every validator runs and the owner hosts |
| **P7** | No single trusted scorer | Every validator runs every model itself; scoring is deterministic; the published figure is stake weighted consensus |

How this compares with existing practice:

| | Public benchmarks | Commercial detection APIs | Kersta |
|---|---|---|---|
| Test exposure | public, contaminated over time | private, unverifiable | hidden, replaced every competition, locked on chain |
| Test conditions | mostly clean files | not disclosed | 70% damaged, matched across real and fake |
| New generators | fixed list | not disclosed | 20% of fakes in every competition |
| Who runs the model | the author | the vendor | every validator, in a sandbox |
| Reported figure | accuracy or AUC | a score | slice scores with calibration, false alarm rate, p95 latency, location |
| Explanation | rarely | rarely | mask or time segments, scored |
| Incentive to be right | none | commercial | emissions against stake weighted consensus |

## 3. System model

A **lane** is one media type. Its definition fixes the input, the speed limit a detector must meet on the reference GPU, the largest package it may submit and the form its location answer takes. A miner may enter one, two or all three lanes, and each lane earns one third of miner emissions.

A **submission** is one Hugging Face repository at one revision, with one folder per lane entered. A lane folder holds `model_config.yaml`, `model.py` and `model.safetensors` ([package_spec.md](docs/package_spec.md)). A **detector** takes one file and returns up to three answers.

| Output | Format | Scored as | Status |
|---|---|---|---|
| Is it fake? | `p_fake`, a probability from 0 to 1 | **Detection** D: MCC and calibration on the clean, damaged and unseen slices | required |
| Where is it fake? | a pixel mask for images; time segments for audio and video | **Location** L: overlap with the true edit on partial fakes | optional |
| What made it? | tool family, or unknown | Attribution | from Phase 3 |

The validator decodes and resizes every file according to the package's `preprocessing` block, so every model sees the same input and model code only ever sees arrays.

## 4. Competition protocol

A competition is one measurement under fixed conditions. The owner decides when each competition runs and announces it on chain from the owner hotkey. Everything that follows is timed by block height, not by the clock. Three blocks separate the phases, and each removes the ability of one party to act on information it has not yet seen.

1. **Announce.** The owner commits `kc1|<n>|<deadline>|<reveal>|<end>|-`, naming the competition's three blocks. Miners read it with `kersta competition`.
2. **Commit.** In any block before the deadline block, the miner commits `ks1|<repo>|<revision>|<digest32>` from its hotkey. The commitment stays in force until it is replaced, so an unchanged detector enters every competition that follows.
3. **Lock.** Before the deadline block, the owner replaces the announcement with the lock `kc1|<n>|<deadline>|<reveal>|<end>|<digest32>`, which carries the digest of the competition's signed test set manifest.
4. **Deadline.** Validators read every commitment as it stood after the block before the deadline block, so every validator derives the same roster, and the same lock, from the chain itself. A commitment written in or after the deadline block enters the next competition.
5. **Upload window.** Between the deadline block and the reveal block, miners make the committed revision public. The files must match the digest.
6. **Reveal.** At the reveal block the test files are released to validators. Their hashes have been on chain since before the deadline.
7. **Scoring.** Between the reveal block and the end block, every validator runs and scores every model, then signs a report.
8. **Weights.** At the end block validators set weights, and re-send them every chain epoch until the next competition ends. The winner of each lane runs in shadow beside the live model before it goes live.
9. **Release.** After the end block, the competition's test set is published as training data.

The owner chooses the gaps between the three blocks. The tools default to an upload window of 1,800 blocks and a scoring window of 3,600 blocks.

## 5. Evaluation

Validators do the expensive computation themselves. There is no trusted scoring service: the owner builds the test set but never scores, and a validator needs only the chain, the test set store and Hugging Face.

**Snapshot.** The validator reads every commitment as it stood just before the deadline block, from an archive node if its own node has pruned that state, and keeps the snapshot.

**Test set integrity.** In the same snapshot it finds the owner's lock. It accepts the manifest only if it is signed by the owner hotkey and its digest matches the lock, then checks the sha256 of every listed file.

**Package integrity.** It lists the repository at the committed revision and refuses forbidden file types and oversized lanes before downloading anything. It downloads only the lane folders, re-runs every package check, and requires the recomputed digest to start with the 32 hex characters on chain.

**Sandbox.** Each lane package runs in a container with no network, a read only package and root filesystem, every capability dropped, an unprivileged user and process and memory limits. Inside, an import guard and an audit hook refuse imports off the allowed list, network use, process creation and file writes outside `/tmp`. Weights load from safetensors only.

**Timing.** Items are sent one at a time, which is also the latency a call or meeting plugin needs. Each answer is timed by the validator, outside the container, after one warm up call; the model cannot reach the clock. Labels never enter the container.

**Copies.** Within a lane, each model is compared with every model committed before it. The earliest commit on chain wins.

**Determinism.** The deadline block, the snapshot, the scoring and the quantisation of weights are deterministic, so honest validators produce the same weight vector. Small differences in GPU arithmetic can move an individual `p_fake` slightly; rank based payout and Yuma consensus absorb them.

## 6. Scoring

Every validator scores with the same code (`kersta/scoring`), using only the outputs it recorded itself. A detector calls an item fake when its `p_fake` is 0.5 or higher.

**Slices.** The test set is divided three ways:

- *clean*: items with no damage;
- *damaged*: items that went through a damage chain;
- *unseen*: fakes from generators no earlier competition used, with the same number of real items.

**Slice score.** Two things are measured on each slice:

- *Correctness*: the Matthews correlation coefficient of the fake or real calls, rescaled from the range −1 to 1 onto 0 to 1. It is taken as 0 when undefined.
- *Calibration*: 1 minus the Brier score (the mean squared gap between `p_fake` and the true label) divided by 0.25, never below 0. A detector that always answers 0.5 has a calibration of 0.

The slice score is the square root of correctness times calibration, so a detector must be both right and honest about its confidence.

**Detection score (D).** The damaged slice counts for half, the clean and unseen slices for a quarter each.

**Location score (L).** Only partial edits count. For each one, the region the detector marks as fake is compared with the true region: the overlap divided by the area the two cover together. Regions are pixels for images, on the model's own input grid, and seconds for audio and video. A partial edit with no location answer scores 0, and L is the average over all partial edits.

**Lane score (S).** Detection counts for 85% and location for 15%. Four checks then apply:

| Check | Effect |
|---|---|
| More than 1% of items failed (crash, timeout, malformed output); below that, a failed item is scored as if `p_fake` were 0.5 | lane score set to 0 |
| 95th percentile time per item above the lane limit | lane score set to 0 |
| More than 2% of real items called fake | lane score halved |
| Copy of an earlier model, as below | lane score set to 0 |

**Copy check.** A model counts as a copy of one committed earlier in the same lane when their `p_fake` values over the whole test set have a Pearson correlation above 0.98 and it does not beat that model's lane score by at least 0.01. Two lane folders with the same digest are always copies.

Every report carries each model's accuracy by damage operation and by generator, so a miner can see exactly which conditions it fails under.

## 7. Incentives

Each lane receives one third of miner emissions, and the 20 best eligible detectors in a lane are paid. First place gets the largest share, each later place gets 0.8 times the place above it, and the shares are scaled to fill the lane. With twenty paid, first place receives about 20.2% of the lane, fifth about 8.3% and tenth about 2.7%.

A hotkey's share of a lane blends 70% of its share from this competition with 30% of its share from the previous one. One strong result therefore cannot take a lane, and one weak one does not erase a record.

Whatever nobody earned is burned to the owner hotkey rather than paid to weak work:
- a lane with no eligible detector;
- the unpaid part of a smoothed lane;
- the share of a hotkey that has left the subnet.

Miners agree on registration that winning models may be used in the Kersta product. Customer revenue funds hosting and buys back the subnet token.

## 8. Threat model

Every known attack maps to a check that any validator can run on its own.

| Attack | Defence |
|---|---|
| Training on the test | Test set new for every competition, locked by hash, revealed after the deadline, used once |
| Learning that damaged means fake | Real and fake damaged at the same rate, by chains drawn without the label |
| Learning file format, size or clip length instead of fakes | One container per lane, metadata stripped, capped size, clip lengths drawn the same way for real and fake |
| Overfitting to known generators | 20% of fakes from generators never used before, scored as a slice |
| Changing the model after seeing results | Package digest committed before the deadline; validators run that exact revision |
| Committing after the deadline | Roster read as of the block before the deadline |
| The test set changed after miners commit | Digest on chain before the deadline, owner signature, sha256 of every file |
| Copying another miner | Correlation check; a copy must beat the original by 0.01; earliest commit wins |
| Malicious code in `model.py` | Container with no network, static code policy, import guard, audit hook, safetensors only |
| Reading the labels | Labels never enter the container; items arrive one at a time, in random order |
| Faking speed | Timed by the validator outside the container |
| Flagging everything as fake | False alarm check on real media; calibration is half of every slice |
| Lazy or dishonest validators | One public test set, deterministic scoring, signed reports anyone can recompute after release, Yuma consensus |

## 9. Participating

| Role | Hardware | What you do | Guide |
|---|---|---|---|
| **Miner** | One 24 GB GPU for training | Train a detector for one or more lanes, publish it on Hugging Face, commit its digest | [docs/miner_setup.md](docs/miner_setup.md) · [docs/miner.md](docs/miner.md) |
| **Validator** | One 48 GB GPU, 16 cores, 64 GB RAM, 1 TB NVMe | Run and score every submission in every competition and set weights | [docs/validator_setup.md](docs/validator_setup.md) |
| **Owner** | Data pipeline and hosting | Announce competitions; build, lock and release each test set; host the winners | [docs/test_sets.md](docs/test_sets.md) |

```bash
git clone https://github.com/kersta-ai/kersta-subnet.git && cd kersta-subnet
pip install -e ".[chain]"

# check a detector the way validators will
kersta miner check submission/
```

Register on testnet with `btcli subnet register --netuid 456 --subtensor.network test`. `kersta competition` prints the competition the owner has announced and its blocks.

## 10. Lanes

| Lane | Input → output | Speed limit (p95) | Max package | Status |
|---|---|---|---|---|
| `image` | one picture → `p_fake`, pixel mask | 100 ms | 2 GiB | testnet, Phase 1 |
| `audio` | a clip of up to 10 s, phone quality included → `p_fake`, time segments | 300 ms | 500 MiB | testnet, Phase 1 |
| `video` | a clip of up to 10 s → `p_fake`, time segments | 2 s | 4 GiB | supported; test sets from Phase 3 |

## 11. Repository layout

```text
kersta/
├── core/         lanes, package spec, outputs, test set format, competition blocks, hashing, constants
├── chain/        commitments, metagraph, weights, the SDK adapter, the offline chain, wallets
├── scoring/      metrics, lane score, copy check, payout
├── media/        preprocessing for models, and the damage tool
├── sandbox/      code policy, worker, runner, wire protocol
├── validator/    fetcher, test sets, inputs, evaluation, competition, weights, reports, state, loop
├── miner/        check, local scoring, upload and commit
├── owner/        test set factory, sign, lock, publish
├── starter/      starter detectors per lane, with training scripts
└── cli/          the `kersta` command
docker/           sandbox and validator images, compose file
docs/             guides for each role
tests/            a hand worked test for every formula, and full competitions on an offline chain
```

`python -m pytest` runs the suite without a GPU, torch, ffmpeg or a chain. Scoring and payout constants live in `kersta/core/constants.py` and cannot be overridden by environment, because validators on different values would lose consensus. A rule change raises `MECHANISM_VERSION`, which is also the on chain weights version key.

## 12. Citation and license

```bibtex
@misc{kersta2026,
  title        = {Kersta: Deepfake Detectors Measured on Hidden, Damaged Media},
  author       = {{Kersta}},
  year         = {2026},
  howpublished = {Bittensor testnet, netuid 456}
}
```

The license is to be decided before public launch.
