"""Explicit image pairing and a pre-training check for exact split overlap."""
import hashlib
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_map(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def paired_maps(directories, expected=None):
    maps = [image_map(directory) for directory in directories]
    common, union = set.intersection(*(set(items) for items in maps)), set.union(*(set(items) for items in maps))
    if common != union:
        raise ValueError(f"Unpaired image IDs: {sorted(union - common)[:5]}")
    if not common:
        raise ValueError("The image split is empty")
    if expected is not None and len(common) != expected:
        raise ValueError(f"Expected {expected} image pairs, found {len(common)}")
    return sorted(common), maps


def decoded_image_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def validate_training_split(cfg, dataset):
    """Check clean-source pixels across splits; never add test samples to training."""
    root = Path(cfg["paths"]["data_dir"])
    name = cfg["data"]["dataset_name"]
    if name == "msrs":
        test_dirs = [root / "test" / "clear" / modality for modality in ("ir", "vi")]
        train_dirs = [Path(dataset.c_ir_dir), Path(dataset.c_vi_dir)]
    elif name == "llvip":
        test_dirs = [root / "clear" / "test" / modality for modality in ("ir", "vi")]
        train_dirs = [Path(dataset.c_ir_dir), Path(dataset.c_vi_dir)]
    elif name == "harvard":
        test_dirs = [root / "clear" / "test" / modality for modality in ("ir", "vi")]
        train_dirs = [Path(dataset.clear_a_dir), Path(dataset.clear_b_dir)]
    else:
        raise ValueError(f"Unknown dataset for split validation: {name}")
    _, test_maps = paired_maps(test_dirs)
    test_hashes = {}
    for mapping in test_maps:
        for path in mapping.values():
            test_hashes.setdefault(decoded_image_hash(path), str(path))
    overlaps = []
    for directory in train_dirs:
        for sample_id in dataset.ids:
            path = directory / sample_id
            digest = decoded_image_hash(path)
            if digest in test_hashes:
                overlaps.append((str(path), test_hashes[digest]))
    if overlaps:
        raise ValueError(f"Training/test clean-image overlap detected: {overlaps[:3]}")
    return {"check": "decoded_RGB_SHA256", "train_pairs": len(dataset),
            "test_unique_images": len(test_hashes), "exact_overlap_images": 0}
