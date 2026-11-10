# LFSDFusion

This release provides inference code for the nine supplied checkpoints
(Harvard, MSRS, and LLVIP; seeds 42, 123, and 3407).
The checkpoint-compatible network and legacy image/text preprocessing are retained.
See [COMPATIBILITY.md](COMPATIBILITY.md) for the scope of this update and
[CHECKPOINTS.md](CHECKPOINTS.md) for weight provenance and historical metadata.

## Setup

Install the project dependencies in `requirements.txt` or `environment.yml`.
The recorded environment uses Python 3.13, PyTorch 2.11.0 with CUDA 13.0,
torchvision 0.26.0, diffusers 0.38.0, PEFT 0.21.2, and transformers 5.7.0.
Install a mutually compatible PyTorch/torchvision CUDA build for your GPU first.
Third-party SD 2.1-base, SAM ViT-B, and LLaVA weights are not bundled;
see [PRETRAINED.md](PRETRAINED.md). Update `paths` in the YAML configs, or
pass `--pretrained-root` and `--data-dir` at inference time.

Relative paths in YAML files are resolved against this repository directory.
Explicit command-line data/model roots are resolved against the calling directory.

## Test-data layout

```text
data/harvard/
  clear/test/{ir,vi}/
  jnm/test/{ir,vi}/
data/msrs/
  test/clear/{ir,vi}/
  test/jibian+noise+motion/{ir,vi}/
data/llvip_lessnew/
  clear/test/{ir,vi}/
  noisy/test/{ir,vi}/
```

Harvard uses 96 pairs, MSRS uses 361 pairs, and LLVIP uses the 50 IDs in
`llvip_test_ids.txt`. The ID file was restored from the original audit snapshot;
no new test subset was sampled. LLVIP pairing uses stems, allowing `.jpg` inputs
and `.png` references with the same ID. Missing selected IDs raise an error.
Optional Harvard `jnm/test/fenlei/` labels are metadata, not fusion inputs.
Clean testing requires only clean images; degraded testing also requires degraded inputs.

## Run one checkpoint

```bash
python test_medical.py --condition degraded --config configs/harvard.yaml --checkpoint checkpoints/harvard/seed42/model_6000.pkl --out-root output/harvard/seed42
python test_msrs.py --condition degraded --config configs/msrs.yaml --checkpoint checkpoints/msrs/seed42/model_6000.pkl --out-root output/msrs/seed42
python test_llvip.py --condition degraded --config configs/llvip_lessnew_test.yaml --checkpoint checkpoints/llvip/seed42/model_6000.pkl --out-root output/llvip/seed42
```

Use `--condition clean` for the clean-input experiment. Other options:

- `--pretrained-root /path/to/pretrained` and `--data-dir /path/to/dataset` override local paths.
- `--captions /path/to/test_captions.json` supplies existing image-specific test captions and skips loading LLaVA.
- `--max-samples 1` limits a later smoke run; omit it for full evaluation.
- `--skip-precompute` computes SAM features on demand to reduce host-memory use.
- `--save-stages` also exports G1/G2/G3 into a separate directory.
- `--device cpu` is available if the required models fit host memory; default is `cuda:0`.

Caption JSON may be a bare `{image_id: description}` dictionary or contain a
`captions` dictionary. Automatically generated caches record checkpoint and input
hashes, dataset, split, condition, and prompt protocol. Different seeds and
conditions use separate caches. Old caches are used only when explicitly selected
with `--captions`; their image IDs must cover the selected test inputs. A cache
with conflicting provenance metadata is rejected.

The model receives only the selected source images and their textual conditions.
Degraded testing does not feed clear references to the fusion network. Test code
uses evaluation/inference mode and contains no optimizer updates. PNG outputs are
in `clear/` or `noise/`; run metadata is stored beside those image directories.

## Run all nine checkpoints

```bash
python test_all.py --condition degraded --device cuda:0
python test_all.py --condition both --device cuda:0 --pretrained-root /path/to/pretrained --data-root /path/to/data
```

`--data-root` expects the `harvard`, `msrs`, and `llvip_lessnew` subdirectories.
Each dataset/seed runs in a fresh process and writes to its own output directory.
`--list` prints commands without executing inference. If supplying existing
captions, `--captions-root` expects `DATASET/seedSEED/test_CONDITION.json`.

## Training split boundaries

Current example configs declare the manuscript splits: Harvard 224 train / 96 test,
MSRS 1083 train / 361 test, and LLVIP 12025 train / an independent 50-pair test subset.
These are current loader/config targets, not edits to the old checkpoint metadata.

Training directories:

```text
Harvard: clear/train/{0,1}, jnm/train/{0,1}
MSRS:   train/{ir,vi}, train/jibian+noise+motion/{ir,vi}
LLVIP:  clear/train/{ir,vi}, jibian+noise+motion/train/{ir,vi}
```

The loaders never concatenate train and test pools. Before training, the entry
point checks decoded RGB hashes against the clean test images and stops on exact
content overlap. This catches identical content despite renamed files or lossless
re-encoding; it does not establish patient-level independence or detect all near
duplicates. The caller must provide the actual independent dataset split.
Training captions use train inputs only and are isolated by seed. Frozen shared
caption files are disabled. A resume checkpoint must carry the same training-input
hash and split-audit metadata; this rule does not restrict inference with old weights.

Example future training uses `train.py --config configs/harvard.yaml --data harvard`
(or `msrs` / `llvip` with matching config). It retains the legacy network required
by the supplied checkpoints; it is not a rewritten implementation of every formula
in `main.tex`. No training or full end-to-end retest was performed in this update.

## Evaluation and degradation generation

`metrics.py` retains its historical evaluation definitions; this compatibility
update does not change or recalculate published metrics. Keep the same evaluator
and preprocessing when comparing outputs. `test.py` is now an alias for Harvard
inference, so there is no second inline evaluator with a different scale.

The MSRS degradation command must match its loader's directory layout:

```bash
python scripts/degrade_dataset.py --task ivif --clear-root data/msrs/train --out-root data/msrs/train/jibian+noise+motion --seed-base 1000 --misreg rigid
```

The existing degradation primitives are retained for checkpoint compatibility;
the Gaussian variance/std mismatch identified in the audit is not represented as
fixed by this inference-only update. Do not regenerate benchmark inputs under a
different degradation protocol and assume the old scores remain applicable.
