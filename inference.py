"""Inference entry point that preserves the supplied checkpoints' architecture."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from runtime import PROJECT_ROOT, dataset_input_fingerprint, load_config, project_path, require_pretrained

CONFIGS = {"harvard": "configs/harvard.yaml", "msrs": "configs/msrs.yaml",
           "llvip": "configs/llvip_lessnew_test.yaml"}
BASE_SEED = 20261029


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_dataset(name, cfg, condition):
    if name == "harvard":
        from data import HarvardConditionSet
        return HarvardConditionSet(cfg, condition)
    if name == "msrs":
        from data_msrs import MSRSTestSet
        return MSRSTestSet(cfg, condition)
    if name == "llvip":
        from data_llvip2s import LLVIP2STestSet
        return LLVIP2STestSet(cfg, condition)
    raise ValueError(f"Unknown dataset: {name}")


def read_captions(path, ids):
    with Path(path).open(encoding="utf-8") as stream:
        saved = json.load(stream)
    captions = saved.get("captions", saved)
    missing = [sample_id for sample_id in ids
               if not isinstance(captions.get(sample_id), str) or not captions[sample_id].strip()]
    if missing:
        raise ValueError(f"Missing or empty captions in {path}: {missing[:5]}")
    return captions, saved.get("meta")


def generate_captions(model, dataset, device, cfg, domain):
    import torch
    from prompts import build_triptych, describe, load_llava
    from tqdm.auto import tqdm

    model.eval()
    panels = {}
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc="Building test triptychs"):
            item = dataset[index]
            first = item["src1"].unsqueeze(0).to(device)
            second = item["src2"].unsqueeze(0).to(device)
            initial = model.run_ftscm(first, second, ids=[item["id"]])
            panels[item["id"]] = build_triptych(item["src1"], item["src2"], initial[0].cpu())
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    llava = processor = None
    try:
        llava, processor = load_llava(cfg["paths"]["llava_dir"],
                                      cfg["prompt"].get("llava_load_in_8bit", False), device=device)
        captions = {}
        for sample_id, panel in tqdm(panels.items(), desc="LLaVA describing"):
            try:
                # Keep the original prompt protocol for compatibility with these checkpoints.
                caption = describe(processor, llava, panel, "IR", "VI", domain=domain)
                if not caption.strip():
                    raise ValueError("Empty LLaVA description")
                captions[sample_id] = caption
            except Exception as error:
                print(f"LLaVA failed for {sample_id}: {error}; using a generic caption")
                captions[sample_id] = "a high-quality fused image"
        return captions
    finally:
        del llava, processor, panels
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(device).eval()


def run_test(name, args):
    import numpy as np
    import torch
    from checkpoint_utils import load_checkpoint
    from tqdm.auto import tqdm
    from torchvision import transforms

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA PyTorch build or pass --device cpu.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = load_config(args.config, args.data_dir, args.pretrained_root)
    dataset = make_dataset(name, cfg, args.condition)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset.ids = dataset.ids[:args.max_samples]
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model", checkpoint)
    for key in ("lora_rank_unet", "lora_rank_vae"):
        if key in state:
            cfg["model"][key] = int(state[key])
    meta = {"checkpoint_sha256": file_sha256(checkpoint_path),
            "inputs_sha256": dataset_input_fingerprint(dataset),
            "dataset": name, "split": "test", "condition": args.condition,
            "domain": "medical" if name == "harvard" else "scene",
            "protocol": "supplied-checkpoints-legacy-v1",
            "prompt_code_sha256": file_sha256(PROJECT_ROOT / "prompts.py"),
            "llava_dir": cfg["paths"]["llava_dir"],
            "llava_8bit": cfg["prompt"].get("llava_load_in_8bit", False)}
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()
    cache_dir = Path(cfg["prompt"]["test_captions_dir"])
    cache_path = cache_dir / f"test_{args.condition}_{key[:16]}.json"
    captions = None
    use_text = cfg["prompt"].get("use_text", True)
    wrong_text = cfg["prompt"].get("wrong_prompt")
    if use_text and not wrong_text:
        if args.captions:
            captions, supplied_meta = read_captions(Path(args.captions).expanduser().resolve(), dataset.ids)
            if supplied_meta is not None and supplied_meta != meta:
                raise ValueError("The explicit caption cache belongs to a different model/input protocol")
            print(f"Using explicitly supplied captions: {args.captions}")
        elif cache_path.exists():
            cached, saved_meta = read_captions(cache_path, dataset.ids)
            if saved_meta == meta:
                captions = cached
    require_pretrained(cfg, needs_llava=(use_text and not wrong_text and captions is None))
    from model import LFSDFusion
    model = LFSDFusion(cfg).to(device)
    model.load_ckpt_from_state_dict(state)
    model.eval()
    del checkpoint, state
    if cfg["training"].get("sam_precompute", True) and not args.skip_precompute:
        model.precompute_sam_features(dataset, device)
    if use_text and not wrong_text and captions is None:
        captions = generate_captions(model, dataset, device, cfg, meta["domain"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"captions": captions, "meta": meta},
                                         ensure_ascii=True, indent=2), encoding="utf-8")
    out_root = project_path(args.out_root or f"output/{name}_test")
    label = "clear" if args.condition == "clean" else "noise"
    output = out_root / label
    stages = out_root / (label + "_stages")
    output.mkdir(parents=True, exist_ok=True)
    if args.save_stages:
        stages.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    with torch.inference_mode():
        for index in tqdm(range(len(dataset)), desc=f"{name}/{args.condition}"):
            random.seed(BASE_SEED + index)
            np.random.seed(BASE_SEED + index)
            torch.manual_seed(BASE_SEED + index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(BASE_SEED + index)
            item = dataset[index]
            first = item["src1"].unsqueeze(0).to(device)
            second = item["src2"].unsqueeze(0).to(device)
            if wrong_text:
                caption = wrong_text + ", high-quality fused image"
            elif use_text:
                caption = captions[item["id"]] + ", high-quality fused image"
            else:
                caption = ""
            result = model(first, second, prompt=[caption], return_all=True, ids=[item["id"]])
            stem = Path(item["id"]).stem
            to_pil((result["g3"][0].cpu() * 0.5 + 0.5).clamp(0, 1)).save(output / (stem + ".png"))
            if args.save_stages:
                for stage in ("g1", "g2", "g3"):
                    image = to_pil((result[stage][0].cpu() * 0.5 + 0.5).clamp(0, 1))
                    image.save(stages / f"{stem}_{stage}.png")
    record = {**meta, "checkpoint": str(checkpoint_path), "sample_ids": dataset.ids,
              "sample_count": len(dataset), "base_seed": BASE_SEED,
              "caption_file": str(args.captions or cache_path) if use_text and not wrong_text else None}
    (out_root / f"{label}_metadata.json").write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved {len(dataset)} fused images to {output}")


def main(name):
    parser = argparse.ArgumentParser(description="Test the supplied legacy LFSDFusion checkpoints")
    parser.add_argument("--condition", choices=("clean", "degraded"), default="degraded")
    parser.add_argument("--config", default=CONFIGS[name])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", help="Override the dataset directory")
    parser.add_argument("--pretrained-root", help="Override the pretrained model root")
    parser.add_argument("--captions", help="Explicit JSON caption file; skips LLaVA generation")
    parser.add_argument("--max-samples", type=int, help="Limit output to the first N test samples")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--save-stages", action="store_true")
    args = parser.parse_args()
    try:
        run_test(name, args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
