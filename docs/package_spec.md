# Detector package, version 1

A detector package is what a miner submits, what every validator runs and what the owner hosts. Every miner, every validator and the hosting service use the format on this page. The version is `1`. It is part of `MECHANISM_VERSION`, so a change to this page is a change to the rules and ships as a new version.

Code: `kersta/core/package.py`, `kersta/media/preprocess.py`, `kersta/core/outputs.py`, `kersta/sandbox/worker.py`.

## Layout

A submission is one Hugging Face model repository at one git revision, with one folder per lane entered:

```text
image/                      any subset of image/, video/, audio/
  model_config.yaml         name, version, lane, dtype, preprocessing
  model.py                  load_model(weights_path, config) -> detector
  model.safetensors         weights; safetensors only
  *.py, *.json, *.yaml      optional helper modules and configs
audio/
  ...
```

Only files inside lane folders belong to the submission. README files, `.gitattributes`, hidden files and everything else in the repository are ignored. They are never downloaded and never hashed.

| Rule | Value |
|---|---|
| Allowed file types | `.py .yaml .yml .json .txt .md .safetensors` |
| Refused | `.bin .pt .pth .pkl .onnx` and every other type, since they can carry a pickle |
| Files per lane | at most 64 |
| Lane folder size | image 2 GiB, video 4 GiB, audio 500 MiB |

## The digest

The submission digest is the sha256 of a listing of every lane file, one line per file, sorted by path:

```text
<path relative to the repository root>\t<sha256 of the file>\t<size in bytes>\n
```

Hashing the listing lets a validator refuse a repository from its file list before downloading it, and keeps two different layouts from producing the same digest. The first 32 hex characters go on chain. `kersta miner hash` prints the digest and a digest per lane.

## model_config.yaml

```yaml
name: my-image-detector          # required, at most 64 characters
version: "1.2.0"                 # required, at most 32 characters
lane: image                      # image | video | audio; must match the folder
dtype: float16                   # float32 | float16 | bfloat16 (also fp32, fp16, bf16)
preprocessing:                   # what the validator turns each file into
  size: [256, 256]
  fit: center_crop
# any other key is passed to load_model unchanged
```

## What your model sees

The validator decodes and resizes every file. Model code never parses a media file.

### Image

| Key | Values | Default |
|---|---|---|
| `size` | `[height, width]`, 16 to 2048 | `[224, 224]` |
| `fit` | `stretch`; `center_crop` (scale to cover, crop the middle); `letterbox` (fit inside, pad black) | `stretch` |
| `interpolation` | `nearest`, `bilinear`, `bicubic`, `lanczos` | `bilinear` |
| `layout` | `nchw`, `nhwc` | `nchw` |
| `scale` | `uint8` (raw 0..255), `unit` (float 0..1), `byte` (float 0..255) | `uint8` |
| `mean`, `std` | three floats each, only with `unit` or `byte` | none |

`batch["inputs"]` is `[B, 3, H, W]` or `[B, H, W, 3]`, `uint8` by default. Cast and normalise on the GPU.

### Audio

| Key | Values | Default |
|---|---|---|
| `sample_rate` | 8000 to 48000 | 16000 |
| `seconds` | 0.5 to 10 | 10 |
| `crop` | `start` or `center`, for clips longer than `seconds` | `start` |
| `mono` | must be true | true |

`batch["inputs"]` is float32 `[B, T]` in [-1, 1], zero padded at the end. `batch["meta"][i]` carries:

| Field | Meaning |
|---|---|
| `samples` | how many samples are real audio |
| `seconds` | the full clip length |
| `offset` | where your window starts in the clip, in seconds |

Return segments in seconds of the original clip, which means adding `offset`. Test set audio is stored as 16 kHz PCM WAV after any codec damage, so decoding is identical on every validator.

### Video

| Key | Values | Default |
|---|---|---|
| `fps` | frames sampled per second, 0.5 to 30 | 4 |
| `max_frames` | 1 to 300 | `fps` × 10 |
| `size`, `fit`, `interpolation`, `layout`, `scale`, `mean`, `std` | as for images, applied to every frame | as for images |
| `audio` | also pass the soundtrack | false |
| `sample_rate` | soundtrack rate | 16000 |

`batch["inputs"]` is `[B, F, 3, H, W]` or `[B, F, H, W, 3]`, with zero frames after the real ones. `meta` carries `frames` (how many are real), `frame_times` (the time of each real frame, in seconds) and `seconds`. With `audio: true`, `batch["audio"]` is float32 `[B, 10 × sample_rate]`.

## The interface

```python
def load_model(weights_path: str, config: dict):
    """Called once. weights_path is <lane>/model.safetensors; config is the parsed YAML."""
    return Detector(...)

class Detector:
    def predict(self, batch: dict) -> list[dict]:
        """batch = {"inputs": array, "meta": [dict], ("audio": array)}; one dict back per item."""
```

Validators send one item per call, so `B = 1`, and time each call. The batch dimension is kept so the same model can batch in hosting.

## What your model returns

| Key | Type | Rule |
|---|---|---|
| `p_fake` | float in [0, 1] | required in every lane |
| `mask` | image: `H × W` in [0, 1], on the input grid | optional; scored as location |
| `segments` | audio and video: `[[start_s, end_s], ...]`, at most 64 | optional; scored as location |
| `source` | tool family name or `"unknown"` | optional; scored from Phase 3 |

Torch tensors, numpy arrays and Python numbers are all accepted. A malformed answer fails that item and never stops the run.

## What model code may use

Every `.py` file is parsed before it runs, and the worker enforces the same list at run time. In the sandbox there is no network, the root filesystem and the package are read only, and writes are refused outside `/tmp`.

**Allowed imports:**

| Group | Modules |
|---|---|
| Standard library | `__future__`, `abc`, `cmath`, `collections`, `contextlib`, `copy`, `dataclasses`, `enum`, `functools`, `itertools`, `json`, `math`, `numbers`, `operator`, `random`, `re`, `statistics`, `string`, `time`, `typing`, `typing_extensions`, `warnings`, `os.path` |
| Libraries | `numpy`, `scipy`, `torch`, `torchvision`, `torchaudio`, `timm`, `einops`, `safetensors`, `transformers`, `PIL` |
| Your package | the other `.py` files in it, absolute or relative |

**Refused:**

- the calls `eval`, `exec`, `compile`, `__import__`, `globals`, `locals`, `vars`, `open`, `breakpoint` and `input`;
- introspection attributes such as `__subclasses__`, `__globals__`, `__builtins__`, and frame attributes;
- `torch.load`, `torch.hub` and `np.load`;
- the keyword arguments `allow_pickle=True`, `trust_remote_code=True` and `weights_only=False`.

Load weights with `safetensors`. `kersta miner check` prints exactly what a validator would refuse.

## Alternative interface: a plain classifier

An existing detector does not need rewriting. `load_model` may return a plain torch module that takes the batch input and returns class logits with class 0 = real. The worker wraps it so that `p_fake = 1 − p(real)`, or uses a sigmoid when there is a single logit. Logits may come back as a tensor, a `.logits` attribute, or a dict with `logits` or `output`.

These alternative config keys are read as follows:

| Key | Read as |
|---|---|
| `modality: image` | `lane: image` |
| `preprocessing.resize: [224, 224]` | `size`, with `fit: center_crop` |
| `num_frames`, `frame_rate` | `max_frames`, `fps` |
| `duration_seconds: 6.0` | `seconds`, centre cropped |
| `model: {num_classes: 3, ...}` | `load_model(weights_path, **model)` is called, and `model.weights_file` names the weights |

A plain classifier gives no location, so it is capped at 0.85 × D until it returns a mask or segments.
