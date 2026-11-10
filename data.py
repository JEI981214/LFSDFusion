import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset_utils import paired_maps

IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                      # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _sorted_names(dir_path):
    return sorted(os.listdir(dir_path))


class PairedHarvardDataset(Dataset):
    def __init__(self, cfg, split="train"):
        data_dir = cfg["paths"]["data_dir"]
        self.split = split
        self.size = cfg["data"]["image_size"]

        jnm_dir = os.path.join(data_dir, "jnm", split)
        clear_dir = os.path.join(data_dir, "clear", split)
        if split == "train":
            self.sub_a, self.sub_b = "0", "1"
        else:
            self.sub_a, self.sub_b = "ir", "vi"

        names_a = set(_sorted_names(os.path.join(jnm_dir, self.sub_a)))
        names_b = set(_sorted_names(os.path.join(jnm_dir, self.sub_b)))
        names_clear_a = set(_sorted_names(os.path.join(clear_dir, self.sub_a)))
        names_clear_b = set(_sorted_names(os.path.join(clear_dir, self.sub_b)))
        common = names_a & names_b & names_clear_a & names_clear_b
        missing = (names_a | names_b | names_clear_a | names_clear_b) - common
        if missing:
            raise RuntimeError(f"Unpaired samples: {len(missing)} items: {sorted(missing)[:5]} ...")
        self.ids = sorted(common)

        self.jnm_a_dir = os.path.join(jnm_dir, self.sub_a)
        self.jnm_b_dir = os.path.join(jnm_dir, self.sub_b)
        self.clear_a_dir = os.path.join(clear_dir, self.sub_a)
        self.clear_b_dir = os.path.join(clear_dir, self.sub_b)

        self.modality_labels = self._build_modality_labels(data_dir)

        expected = cfg["data"]["expected_" + ("train_pairs" if split == "train" else "test_pairs")]
        if len(self.ids) != expected:
            raise ValueError(f"{split} expected {expected} pairs; got {len(self.ids)}")

    def _build_modality_labels(self, data_dir):

        if self.split == "train":
            return {i: {"A": "A", "B": "B"} for i in self.ids}

        labels = {}
        fenlei_dir = os.path.join(data_dir, "jnm", self.split, "fenlei")
        for combo in os.listdir(fenlei_dir):
            for sub, tag in (("0", "A"), ("1", "B")):
                sub_dir = os.path.join(fenlei_dir, combo, sub)
                if not os.path.isdir(sub_dir):
                    continue
                for name in os.listdir(sub_dir):
                    labels.setdefault(name, {})[tag] = combo
        missing = [i for i in self.ids if i not in labels or "A" not in labels[i] or "B" not in labels[i]]
        if missing:
            raise RuntimeError(f"Missing modality labels in fenlei: {missing[:5]} ...")
        return labels

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        return IMG_TRANSFORM(img)

    def __getitem__(self, idx):
        name = self.ids[idx]
        out = {
            "id": name,
            "src1": self._load(os.path.join(self.jnm_a_dir, name)),
            "src2": self._load(os.path.join(self.jnm_b_dir, name)),
            "clear1": self._load(os.path.join(self.clear_a_dir, name)),
            "clear2": self._load(os.path.join(self.clear_b_dir, name)),
            "label1": self.modality_labels[name]["A"],
            "label2": self.modality_labels[name]["B"],
        }
        return out


class HarvardConditionSet(Dataset):
    def __init__(self, cfg, condition):
        self.size = cfg["data"]["image_size"]
        assert condition in ("clean", "degraded")
        clear = os.path.join(cfg["paths"]["data_dir"], "clear", "test")
        base = clear if condition == "clean" else os.path.join(cfg["paths"]["data_dir"], "jnm", "test")
        self.ir_dir = os.path.join(base, "ir")
        self.vi_dir = os.path.join(base, "vi")
        stems, maps = paired_maps(
            [self.ir_dir, self.vi_dir, os.path.join(clear, "ir"), os.path.join(clear, "vi")],
            expected=cfg["data"].get("expected_test_pairs", 96))
        self.ids = [maps[0][stem].name for stem in stems]
        self.input_paths = {maps[0][stem].name: (maps[0][stem], maps[1][stem]) for stem in stems}
        self.labels = {name: {"A": "A", "B": "B"} for name in self.ids}
        fenlei = os.path.join(cfg["paths"]["data_dir"], "jnm", "test", "fenlei")
        if os.path.isdir(fenlei):
            for combo in sorted(os.listdir(fenlei)):
                for sub, tag in (("0", "A"), ("1", "B")):
                    directory = os.path.join(fenlei, combo, sub)
                    if os.path.isdir(directory):
                        for name in os.listdir(directory):
                            if name in self.labels:
                                self.labels[name][tag] = combo

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        return IMG_TRANSFORM(img)

    def __getitem__(self, idx):
        name = self.ids[idx]
        lab = self.labels[name]
        return {
            "id": name,
            "src1": self._load(self.input_paths[name][0]),
            "src2": self._load(self.input_paths[name][1]),
            "label1": lab["A"], "label2": lab["B"],
        }


def collate_fn(batch):
    out = {}
    for k in ["src1", "src2", "clear1", "clear2"]:
        out[k] = torch.stack([b[k] for b in batch])
    for k in ["id", "label1", "label2"]:
        out[k] = [b[k] for b in batch]
    return out
