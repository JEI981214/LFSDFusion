import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from degradations import (gaussian_noise, motion_blur, poisson_noise, rician_noise,
                          rigid_transform, nonrigid_transform, stripe_noise)

POISSON_PEAKS = {"light": 160, "medium": 60, "heavy": 25}
RICIAN_SIGMAS = {"light": 8, "medium": 20, "heavy": 40}
MIF_NOISE_MODALITIES = {"CT": "poisson", "PET": "poisson", "SPECT": "poisson", "MRI": "rician"}


def _load(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _save(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def _degrade_ir(img, seed, misreg):
    rng = np.random.default_rng(int(seed))
    h, w = img.shape[:2]
    low = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    out = stripe_noise(out, 15, rng)
    if misreg == "rigid":
        out = rigid_transform(out, rng, theta_max=7.0, t_max=0.07)
    elif misreg == "nonrigid":
        out = nonrigid_transform(out, rng)
    elif misreg is not None:
        raise ValueError(misreg)
    return out


def _degrade_vi(img, seed):
    rng = np.random.default_rng(int(seed))
    angle = [0, 45, 90][int(rng.integers(0, 3))]
    out = motion_blur(img, angle, rng)
    return gaussian_noise(out, 15, rng)


def _degrade_medical(img, modality, seed, level):
    rng = np.random.default_rng(int(seed))
    angle = [0, 45, 90][int(rng.integers(0, 3))]
    out = motion_blur(img, angle, rng)
    if MIF_NOISE_MODALITIES[modality] == "poisson":
        out = poisson_noise(out, POISSON_PEAKS[level], rng)
    else:
        out = rician_noise(out, RICIAN_SIGMAS[level], rng)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["ivif", "mif"])
    ap.add_argument("--clear-root", required=True,
                    help="clean source folder containing modality subfolders")
    ap.add_argument("--out-root", required=True,
                    help="output folder (same subfolder layout)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--level", default="light", choices=["light", "medium", "heavy"],
                    help="MIF noise level (IVIF levels are fixed)")
    ap.add_argument("--misreg", default="rigid", choices=["rigid", "nonrigid", "none"],
                    help="IR misregistration type (IVIF only)")
    ap.add_argument("--mod-map", default=None,
                    help='MIF modality mapping for non-modality folder names, e.g. {"0":"CT","1":"MRI"}')
    ap.add_argument("--mods", nargs="+", default=None,
                    help="subfolders to process (default: all subfolders)")
    args = ap.parse_args()

    mod_map = {}
    if args.mod_map:
        import json
        mod_map = json.loads(args.mod_map)

    subdirs = args.mods if args.mods else sorted(
        d for d in os.listdir(args.clear_root)
        if os.path.isdir(os.path.join(args.clear_root, d)))
    for sub in subdirs:
        src_dir = os.path.join(args.clear_root, sub)
        dst_dir = os.path.join(args.out_root, sub)
        modality = mod_map.get(sub, sub)
        for i, name in enumerate(sorted(os.listdir(src_dir))):
            img = _load(os.path.join(src_dir, name))
            seed = args.seed_base + i
            if args.task == "ivif":
                out = _degrade_ir(img, seed, None if args.misreg == "none" else args.misreg)\
                    if sub in ("ir", "IR") else _degrade_vi(img, seed)
            else:
                assert modality in MIF_NOISE_MODALITIES, f"unknown modality {modality} for {sub}"
                out = _degrade_medical(img, modality, seed, args.level)
            _save(os.path.join(dst_dir, name), out)
            print(f"[{args.task}] {sub}/{name} done")
    print(f"degradation done -> {args.out_root}")


if __name__ == "__main__":
    main()
