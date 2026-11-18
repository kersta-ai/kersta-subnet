# Running a Kersta validator

Netuid **456** on Bittensor testnet.

## Requirements

| | |
|---|---|
| Machine | one 48 GB NVIDIA GPU, at least as fast as the published reference GPU; 16 cores; 64 GB RAM; 1 TB NVMe |
| OS | Ubuntu 22.04 or 24.04 |
| Software | Docker 27 or newer, NVIDIA Container Toolkit |
| Network | a stable connection; every competition the loop downloads each committed package and the competition's test set |
| Uptime | the loop must be running from a competition's reveal block until weights are set at its end block, and always on for the weights keeper |
| Access | a hotkey registered on netuid 456 with enough stake for a validator permit |

## 1. Host preparation

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
sudo mkdir -p /var/lib/kersta && sudo chown $USER /var/lib/kersta
```

Install the NVIDIA Container Toolkit and confirm `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` lists the card.

`/var/lib/kersta` has to exist at that exact path on the host. The validator bind mounts it at the same path inside, so the sandbox containers it starts see the same package folders.

## 2. Register the hotkey

```bash
btcli subnet register --netuid 456 --subtensor.network test --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 3. Configure

```bash
git clone https://github.com/kersta-ai/kersta-subnet.git && cd kersta-subnet
cp .env.example .env    # wallet names, KERSTA_TESTSET_URL, KERSTA_GPU
docker pull ghcr.io/kersta-ai/sandbox:1
```

Check every chain read path against the network:

```bash
kersta chain doctor --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

It prints which SDK path answered each call. Run it again after every SDK upgrade.

## 4. Run

```bash
docker compose -f docker/docker-compose.validator.yml up -d
docker compose -f docker/docker-compose.validator.yml logs -f
```

or from a checkout:

```bash
pip install -e ".[chain]"
kersta validator run --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Miner models always run in Docker. A validator on a real network refuses `--sandbox local`.

## The eight steps of a competition

The loop reads block heights, not the host clock. It follows the competition the owner announces on chain and remembers every one it has seen. Once a competition's reveal block has passed it runs these steps once; a competition already reported is not run again.

1. **Snapshot.** Read every commitment as it stood after the block before the deadline block, and keep the snapshot in `state/competitions/<n>/roster.json`.
2. **Verify the test set.** In the same snapshot, the owner's commitment must be the lock for exactly this competition: `kc1|<n>|<deadline>|<reveal>|<end>|<digest32>`. Without it, nothing is downloaded and the competition is not scored. Fetch `competition-<n>/manifest.json` from the store. Accept it only if it is signed by the owner hotkey and its digest starts with the locked prefix, then check the sha256 of every file.
3. **Fetch packages.** For every `ks1` commitment from a hotkey that is still registered:
   - list the repository at the committed revision;
   - refuse forbidden files and oversized lanes before downloading;
   - download the lane folders;
   - re-run every package check;
   - require the digest to match.

   Verified packages are cached by digest.
4. **Decode inputs.** Decode each test file with each package's `preprocessing` block. Decoded inputs are cached, so models asking for the same preprocessing share one decode.
5. **Run.** For each lane package, start the sandbox container. Send one warm up item, then every item one at a time, timing each answer from outside. Labels never enter the container.
6. **Score.** Compute the slice scores, D, L and S, then the failure, speed and false alarm checks. Run the copy check across the lane, rank, and take the top 20 shares.
7. **Pay.** Blend 70/30 with the previous competition's shares and give each lane one third. Burn to the owner hotkey whatever nobody earned. If you have no shares for the previous competition, the loop scores it first, so your smoothing matches every other validator. Its blocks come from your own memory or, if you joined late, from the signed manifest.
8. **Weights and report.** Sign the competition report: every lane score with its breakdown by damage operation and generator, the payout and the weights. Keep per item records for audits. Once the end block has passed, set the weights.

Between competitions, a background keeper sends the same weights again once every chain epoch, so the validator never passes the activity cutoff however far apart competitions are. Weights the chain refused are kept and sent again a few minutes later.

## Dry run

```bash
kersta validator score-competition <n> --dry-run --item-limit 50 --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

Scores fifty items per lane, posts nothing and sets no weights. Use it after installation and after every image update.

## Updating

```bash
docker compose -f docker/docker-compose.validator.yml pull
docker compose -f docker/docker-compose.validator.yml up -d
```

The sandbox image is pulled by the digest pinned in each release, never by tag alone. When the scoring rules change, the release raises `MECHANISM_VERSION`, and the chain refuses weights from validators still on the old rules.

## Weights

Weights come from the payout:
- the top 20 per lane, each rank 0.8 times the rank above;
- each share 70% this competition and 30% the previous one;
- one third per lane;
- burns to the owner hotkey.

They are mapped to UIDs on a freshly read metagraph, and the share of a hotkey that has left the subnet is burned rather than spread over the rest. Then they are quantised to u16 values summing to exactly 65535, by largest remainder with ties to the lower UID, so every honest validator signs the same vector.

The version key is `MECHANISM_VERSION` (1.0.0 is 10000). Commit reveal is left to the SDK, which chooses plain or commit reveal submission from the subnet's hyperparameters.

Every attempt is appended to `state/weights_history.jsonl` with:
- the raw shares;
- the u16 vector;
- the version key;
- the chain's answer.

## Archive node

Commitments are read as they stood before the deadline block. A pruned node keeps only a few hundred blocks of state. The loop takes the snapshot as soon as the reveal block passes, but a validator that starts late needs an archive node. Set `KERSTA_ARCHIVE_ENDPOINT`, or pass `--archive-endpoint`.

## State on disk

```text
/var/lib/kersta/validator/
  state/competitions.json                  every competition seen announced (its blocks)
  state/competitions/<n>/roster.json       commitments at the deadline
  state/competitions/<n>/scores.json       every lane score
  state/competitions/<n>/report.json       the signed report
  state/competitions/<n>/items/*.jsonl.gz  per item outputs, for audits and appeals
  state/shares.json                        each competition's ranking shares, for smoothing
  state/weights.json                       the weights on chain now
  state/weights_history.jsonl              every vector sent and the chain's answer
  cache/                                   verified packages, test sets and decoded inputs
```

Back up `state/`. The cache can be deleted at any time.

## Compute budget

Each model runs one item at a time. The worst case per model is:

| Lane | Worst case per model |
|---|---|
| image | 4,000 × 100 ms ≈ 7 minutes |
| audio | 4,000 × 300 ms ≈ 20 minutes |
| video | 4,000 × 2 s ≈ 2.2 hours |

Real detectors are usually far faster than their limits. A full video lane with many miners does not fit the default scoring window of 3,600 blocks on one GPU; [decisions.md](decisions.md) lists the options before the video lane reaches full weight.
