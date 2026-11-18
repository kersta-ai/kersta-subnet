# Mining on Kersta

A submission is one Hugging Face repository with a folder for each lane you enter. `image/`, `video/` and `audio/` each hold `model_config.yaml`, `model.py` and `model.safetensors`. Validators fetch it at the revision you commit on chain and run every lane on their own GPUs. You host nothing.

Start from a starter detector: `kersta miner init --lane <lane>`. [miner.md](miner.md) walks through training, `kersta miner check`, `kersta miner score`, `kersta miner submit` and `kersta miner reveal`. [package_spec.md](package_spec.md) is the exact contract.

Register a hotkey with `btcli subnet register --netuid 456 --subtensor.network test`.

## Training playbook

Validators score your detector on media it has never seen, 70% of it damaged and 20% of its fakes from generators no earlier competition used. A detector that only learned clean benchmark files will not rank. The order that works:

1. **Baseline.** Score the starter detector on a released competition with `kersta miner score`. Note the slices and the weakest damage operations and generators it reports.
2. **Damage.** Train with the damage tool on 70% of items, real and fake alike (`kersta damage`, or the starter `train.py`). A model that sees damage only on fakes learns that damage means fake.
3. **Generalise.** Hold out whole generators, not random files, when you validate. The unseen slice is a quarter of D.
4. **Calibrate.** Fit a temperature on held out data and fold it into the head. Calibration is half of every slice score.
5. **Locate.** Add a mask head (image) or a per frame head (audio, video), trained on partial edits. Location is 15% of S, and every model without it is capped at 0.85 × D.
6. **Check.**
   - Keep the p95 time per item under the lane limit with margin, because validators use the reference GPU, not yours.
   - Keep false alarms on real media under 2%.
   - Run `kersta miner check` and `kersta miner score` once more.

Items are fresh in every competition, so train the skill, not the generators. Every released competition is free training data: pass it to `train.py --testset`.
