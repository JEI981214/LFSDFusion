"""
  python train.py --config config.yaml --data msrs
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import PairedHarvardDataset, collate_fn
from losses import FusionLoss
from model import LFSDFusion
from dataset_utils import validate_training_split
from runtime import dataset_input_fingerprint, load_config
from prompts import (PromptCache, build_triptychs_for_dataset, describe,
                     ftscm_snapshot_hash, load_llava)

FALLBACK_CAPTION = "a high-quality fused medical image"
FALLBACK_CAPTION_SCENE = "a high-quality fused image"
SCENE_DATASETS = {"msrs", "llvip", "llvip2s"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def refresh_prompt_cache(model, dataset, cache, llava_dir, device, step, cfg, domain="medical"):

    print(f"[step {step}] refreshing LLaVA prompt cache ...")
    model.eval()

    triptych_dir = os.path.join(cache.cache_dir, "triptychs")
    paths = build_triptychs_for_dataset(model, dataset, triptych_dir, device)
    ftscm_hash = ftscm_snapshot_hash(model)


    model.to("cpu")
    torch.cuda.empty_cache()
    llava_model, processor = load_llava(llava_dir, cfg["prompt"].get("llava_load_in_8bit", False))
    captions = {}
    fallback = FALLBACK_CAPTION if domain == "medical" else FALLBACK_CAPTION_SCENE
    for sid, (path, l1, l2) in tqdm(paths.items(), desc="LLaVA describing"):
        from PIL import Image
        try:
            captions[sid] = describe(processor, llava_model, Image.open(path).convert("RGB"),
                                     l1, l2, domain=domain)
        except Exception as e:
            print(f"LLaVA failed for {sid}: {e}")
            captions[sid] = fallback
    del llava_model, processor
    torch.cuda.empty_cache()


    cache.update(captions, ftscm_hash, step, domain=domain,
                 dataset_hash=dataset_input_fingerprint(dataset))
    model.to(device)
    model.train()
    return cache


def get_caption(cache, sample_id, domain="medical"):
    cap = cache.get(sample_id)
    fallback = FALLBACK_CAPTION if domain == "medical" else FALLBACK_CAPTION_SCENE
    return (cap + ", high-quality fused image") if cap else fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/harvard.yaml")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .pkl file")
    parser.add_argument("--data", default="harvard",
                        choices=["harvard", "msrs", "llvip", "harvard2", "llvip2s"],
                        help="Training dataset")
    args = parser.parse_args()
    domain = "scene" if args.data in SCENE_DATASETS else "medical"

    cfg = load_config(args.config)
    dataset_name = {"harvard2": "harvard", "llvip2s": "llvip"}.get(args.data, args.data)
    if cfg["data"].get("dataset_name") != dataset_name:
        raise ValueError("The selected dataset does not match data.dataset_name in the config")
    if cfg["data"].get("train_split") != "train":
        raise ValueError("Training is restricted to the train split")
    if not cfg["data"].get("expected_train_pairs"):
        raise ValueError("A test-only config cannot be used for training")
    if cfg["prompt"].get("shared_captions_file"):
        raise ValueError("Shared frozen captions are disabled; use per-run train-split captions")
    t_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(t_cfg["seed"])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    out_dir = os.path.join(cfg["paths"]["output_dir"], f"seed{t_cfg['seed']}")
    cfg["paths"]["output_dir"] = out_dir
    cfg["prompt"]["cache_dir"] = os.path.join(out_dir, "train_prompt_cache")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)


    if args.data == "msrs":
        from data_msrs import MSRSTrainDataset
        dataset_train = MSRSTrainDataset(cfg, split=cfg["data"]["train_split"])
    elif args.data == "llvip":
        from data_llvip import LLVIPTrainDataset
        dataset_train = LLVIPTrainDataset(cfg, split=cfg["data"]["train_split"])
    elif args.data == "harvard2":
        from data_harvard2 import Harvard2TrainDataset
        dataset_train = Harvard2TrainDataset(cfg, split=cfg["data"]["train_split"])
    elif args.data == "llvip2s":
        from data_llvip2s import LLVIP2STrainDataset
        dataset_train = LLVIP2STrainDataset(cfg, split=cfg["data"]["train_split"])
    else:
        dataset_train = PairedHarvardDataset(cfg, split=cfg["data"]["train_split"])
    dl_train = DataLoader(dataset_train, batch_size=t_cfg["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
                          drop_last=True)

    split_check = validate_training_split(cfg, dataset_train)
    train_input_hash = dataset_input_fingerprint(dataset_train)
    with open(os.path.join(out_dir, "dataset_split_check.json"), "w", encoding="utf-8") as stream:
        json.dump({**split_check, "train_ids": dataset_train.ids,
                   "train_inputs_sha256": train_input_hash}, stream, indent=2)
    print("building model ...")
    model = LFSDFusion(cfg).to(device)
    layers_to_opt = model.trainable_parameters()
    print(f"trainable parameters: {sum(p.numel() for p in layers_to_opt) / 1e6:.3f}M")
    optimizer = torch.optim.AdamW(layers_to_opt, lr=t_cfg["learning_rate"],
                                  betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer,
                                 num_warmup_steps=t_cfg["warmup_steps"],
                                 num_training_steps=t_cfg["max_steps"])
    loss_fn = FusionLoss(cfg).to(device)


    global_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_check = ckpt.get("dataset_split_check")
        if saved_check is None or saved_check.get("train_inputs_sha256") != train_input_hash:
            raise ValueError("Resume checkpoint has no matching split audit; use the same verified training split")
        model.load_ckpt_from_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        for g in optimizer.param_groups:
            g["lr"] = t_cfg["learning_rate"]
        print(f"resumed from {args.resume} at step {global_step}, lr={t_cfg['learning_rate']}")


    use_sam = cfg["model"].get("use_sam", True) and not cfg["model"].get("bdc_variant", False)
    if use_sam and cfg["training"].get("sam_precompute", True):
        model.precompute_sam_features(dataset_train, device)


    use_text = cfg["prompt"].get("use_text", True)
    wrong_text = cfg["prompt"].get("wrong_prompt")
    cache = PromptCache(cfg["prompt"]["cache_dir"])
    if (use_text and not wrong_text
            and (set(cache.cache) != set(dataset_train.ids)
                 or cache.is_stale(ftscm_snapshot_hash(model), domain, train_input_hash))):
        cache = refresh_prompt_cache(model, dataset_train, cache, cfg["paths"]["llava_dir"],
                                     device, global_step, cfg, domain=domain)


    progress_bar = tqdm(range(t_cfg["max_steps"]), initial=global_step, desc="Steps")
    model.train()
    accum = t_cfg["gradient_accumulation_steps"]
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    epoch = start_epoch
    batch_idx = 0
    while global_step < t_cfg["max_steps"]:
        for batch in dl_train:
            x_src1 = batch["src1"].to(device)
            x_src2 = batch["src2"].to(device)
            x_src11 = batch["clear1"].to(device)
            x_src22 = batch["clear2"].to(device)

            if wrong_text:
                captions = [wrong_text + ", high-quality fused image"] * len(batch["id"])
            else:
                captions = [get_caption(cache, sid, domain) for sid in batch["id"]] if use_text else [""] * len(batch["id"])
            out = model(x_src1, x_src2, prompt=captions, return_all=True, ids=batch["id"])
            x_tgt_pred = out["g3"]

            loss, parts = loss_fn(x_tgt_pred, x_src11, x_src22, return_parts=True)
            loss = loss / accum
            loss.backward()
            batch_idx += 1

            if batch_idx % accum == 0:
                torch.nn.utils.clip_grad_norm_(layers_to_opt, t_cfg["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress_bar.update(1)
                global_step += 1

                logs = {k: round(v.detach().item(), 4) for k, v in parts.items()}
                logs["loss"] = round(loss.detach().item() * accum, 4)
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)


                if global_step % t_cfg["checkpoint_every"] == 0:
                    outf = os.path.join(ckpt_dir, f"model_{global_step}.pkl")
                    torch.save({
                        "model": model.save_model(None),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": cfg,
                        "prompt_cache_meta": cache.meta,
                        "dataset_split_check": {**split_check, "train_inputs_sha256": train_input_hash},
                    }, outf)
                    print(f"saved checkpoint {outf}")


                if use_text and not wrong_text\
                        and global_step % cfg["prompt"]["refresh_every_steps"] == 0:
                    cache = refresh_prompt_cache(model, dataset_train, cache,
                                                 cfg["paths"]["llava_dir"], device, global_step, cfg,
                                                 domain=domain)


                if global_step % t_cfg["eval_every"] == 0:
                    model.eval()
                    with torch.no_grad():
                        for j in range(min(8, len(dataset_train))):
                            item = dataset_train[j]
                            src1 = item["src1"].unsqueeze(0).to(device)
                            src2 = item["src2"].unsqueeze(0).to(device)
                            cap = (wrong_text + ", high-quality fused image") if wrong_text else (
                                get_caption(cache, item["id"], domain) if use_text else "")
                            out = model(src1, src2, prompt=[cap], return_all=True, ids=[item["id"]])
                            from PIL import Image
                            from torchvision import transforms
                            for gname in ("g1", "g2", "g3"):
                                pil = transforms.ToPILImage()((out[gname][0].cpu() * 0.5 + 0.5).clamp(0, 1))
                                pil.save(os.path.join(eval_dir, f"step{global_step}_{item['id']}_{gname}.png"))
                    model.train()

                if global_step >= t_cfg["max_steps"]:
                    break
        epoch += 1

    print("training finished")


if __name__ == "__main__":
    main()
