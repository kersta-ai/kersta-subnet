# Building and submitting a detector

Miners build detectors and validators measure them. You commit a package before a competition's deadline block and make it public during the upload window. Every validator then runs it on its own GPU, inside a sandbox, on the competition's hidden test set. Nothing runs on your machine during the competition, and nothing timed on your machine is used for scoring.

## Lanes and limits

| Lane | Input | Speed limit (p95, reference GPU) | Max package | Location answer |
|---|---|---|---|---|
| `image` | one picture | 100 ms | 2 GiB | pixel mask |
| `video` | a clip of up to 10 s | 2 s | 4 GiB | time segments |
| `audio` | a clip of up to 10 s, phone quality included | 300 ms | 500 MiB | time segments |

A miner may enter any subset of lanes with one submission. Each lane earns one third of miner emissions.

## What your machine needs

- **A GPU for training.** One 24 GB card is enough for the starters.
- **Python 3.10 or newer** and the kit:

  ```bash
  pip install -e ".[chain,torch]"
  ```

- **A hotkey** registered on netuid 456:

  ```bash
  btcli subnet register --netuid 456 --subtensor.network test
  ```

- **A Hugging Face account** with a write token (`export HF_TOKEN=...`).
- **ffmpeg**, for the video lane and audio codec damage.

## Start from a starter

```bash
kersta miner init --lane image --out submission
python -m kersta.starter.image.train --sources data/ --out submission/image/model.safetensors
```

`--sources` takes the same layout the test set factory uses:
- `real/`
- `fake/<generator>/`
- `partial/<generator>/` with `.mask.png` or `.segments.json` sidecars.

`--testset` takes any released competition. Training damages 70% of items with the same damage tool validators use, real and fake alike, and fits a temperature on held out data.

To damage a file by hand:

```bash
kersta damage --lane image --in photo.jpg --out damaged.png --chain "social:long_edge=1080,quality=75"
kersta damage --lane audio --in clip.wav --out phone.wav --chain "phone"
```

Without `--chain` the tool draws a random chain, the way the test set factory does.

## Before you commit

Check the package the way validators will:

```bash
kersta miner check submission
```

It prints the digest and each lane's digest and size, or the reason a validator would refuse the package:

- a file type;
- a size;
- a config field;
- a safetensors header;
- a line of code against the policy.

Then it starts your model in a worker process and sends it random inputs of exactly the shape validators send. A model that fails here is excluded at the deadline.

Score it on a released competition with the validators' own code:

```bash
kersta miner score submission --testset released/competition-1/
```

It prints the slice scores, D, L and S, the false alarm rate, the p95 time and your weakest damage operations and generators. The time is measured on your GPU and validators use the reference GPU, so leave margin under the limit.

## The commitment

```bash
kersta miner submit submission --repo <user>/<name> --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

This:
1. uploads the lane folders to a **private** repository, removing stale lane files from earlier uploads in the same commit;
2. confirms that the files at the new revision are exactly the local files;
3. commits from your hotkey:

```text
ks1|<user>/<name>|<revision>|<digest32>
```

The commitment is at most 128 bytes, which leaves 50 characters for `<user>/<name>`. Commit in any block before the competition's deadline block; `kersta competition` prints it.

The commitment stays in force until you replace it, so an unchanged detector is scored again on every new test set. Committing again is always safe. A commitment written in or after a deadline block enters the next competition and does not touch the one being scored.

## The upload window

Between the deadline block and the reveal block, make the committed revision public:

```bash
kersta miner reveal --repo <user>/<name>
```

A repository that is still private when validators fetch it scores nothing in this competition. Keep it private until the deadline, so the files cannot be copied before your commit counts. Never delete or rewrite the committed revision, and never rename the repository; validators download that exact revision.

## How your detector is run

Each validator, for each lane folder in your submission:

1. **Lists the repository** at the committed revision and refuses forbidden file types, more than 64 files, or an oversized lane, before downloading anything.
2. **Downloads only the lane folders**, re-runs every package check, and requires the recomputed digest to start with your 32 hex characters.
3. **Starts your model** in the pinned sandbox image:
   - no network;
   - read only package and root filesystem;
   - every capability dropped;
   - an unprivileged user;
   - process and memory limits.

   `load_model` has 15 minutes.
4. **Sends one warm up item**, whose answer is not scored or timed.
5. **Sends every test item one at a time**, decoded by the validator with your `preprocessing` block. It times each answer from outside the container, with a hard timeout of twenty times the lane limit, and at least 10 seconds.
6. **Validates every answer** against [package_spec.md](package_spec.md) and scores the lane.

A validator stops a run early only when the result is already certain for the whole set: more than 1% of items failed, or more than 5% over the speed limit. The rest are recorded as not run.

## Sandbox rules for model code

Code that breaks a rule is refused before it runs, or fails at the call that broke it.

- **Imports:** only from the allowed list (standard library computation modules, `numpy`, `scipy`, `torch`, `torchvision`, `torchaudio`, `timm`, `einops`, `safetensors`, `transformers`, `PIL`) and your own package's `.py` files.
- **Forbidden calls:** no `eval`, `exec`, `compile`, `__import__`, `open`, `globals`, `locals` or `vars`.
- **No introspection:** no `__subclasses__`, `__globals__`, `__builtins__` or frame attributes.
- **No pickles:** no `torch.load`, `np.load`, `allow_pickle=True`, `trust_remote_code=True` or `weights_only=False`. Load weights with safetensors.
- **No network and no processes.** Both are blocked twice, by the container and by an audit hook in the worker.
- **No writes outside `/tmp`.** The audit hook refuses them.
- **Output:** anything you print goes to stderr. It cannot reach the protocol stream.

## What is checked

| Limit | Enforced before the run | Checked during the run |
|---|---|---|
| File types and count | listing refused before download | none |
| Package size per lane | listing refused before download | none |
| Weight format | safetensors headers parsed without loading | none |
| Model code | static policy on every `.py` file | import guard and audit hook |
| Load time | none | 15 minutes, then the lane fails |
| Speed | none | p95 time per item from outside the container, against the lane limit |
| Item timeout | none | twenty times the lane limit, at least 10 s; the run ends there |
| Output format | none | every answer validated; a malformed one fails the item |
| False alarms | none | share of real items with `p_fake ≥ 0.5`, against 2% |

## Copies

Every validator compares each lane model with every model committed before it in the same lane, ordered by commit block, then hotkey.

| Checked by | A copy when | Effect |
|---|---|---|
| Lane digest | identical files to an earlier lane | scores zero in that lane |
| Outputs | Pearson correlation of `p_fake` above 0.98 over the whole test set, and not 0.01 better than the earlier model | scores zero in that lane |

Your other lanes are still scored. Replacing your commitment moves your commit block to now for every lane in the submission. If you change only one lane, consider entering lanes from separate hotkeys.

## Reading your results

Every validator publishes a signed report for the competition. For each lane, each row carries:
- the slice scores, D, L and S, and the rank and share;
- the false alarm rate and the p95 time;
- accuracy by damage operation and by generator.

`kersta miner status` prints your commitment, the block it was written in and the competition it enters.

## After the competition ends

Weights for the competition are set at its end block and re-sent every chain epoch until the next competition ends. Your share of a lane is 70% this competition's rank weight and 30% the previous one's, so a strong result keeps paying into the next competition. The competition's test set is then published as training data.
