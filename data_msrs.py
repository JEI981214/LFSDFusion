import os

from PIL import Image
from torch.utils.data import Dataset

from data import IMG_TRANSFORM


class MSRSTrainDataset(Dataset):
    def __init__(self, cfg, split="train"):
        if split != "train":
            raise ValueError("MSRS training requires the train split")
        root = cfg["paths"]["data_dir"]
        deg = os.path.join(root, "train", "jibian+noise+motion")
        clear = os.path.join(root, split)

        names = [set(os.listdir(os.path.join(d, m))) for d, m in
                 ((deg, "ir"), (deg, "vi"), (clear, "ir"), (clear, "vi"))]
        common = set.intersection(*names)
        missing = set.union(*names) - common
        if missing:
            raise RuntimeError(f"Unpaired samples: {len(missing)} items: {sorted(missing)[:5]} ...")
        self.ids = sorted(common)

        self.ir_dir = os.path.join(deg, "ir")
        self.vi_dir = os.path.join(deg, "vi")
        self.c_ir_dir = os.path.join(clear, "ir")
        self.c_vi_dir = os.path.join(clear, "vi")

        expected = cfg["data"].get("expected_train_pairs")
        if expected is not None and len(self.ids) != expected:
            raise ValueError(f"train expected {expected} pairs; got {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        return IMG_TRANSFORM(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        name = self.ids[idx]
        return {
            "id": name,
            "src1": self._load(os.path.join(self.ir_dir, name)),
            "src2": self._load(os.path.join(self.vi_dir, name)),
            "clear1": self._load(os.path.join(self.c_ir_dir, name)),
            "clear2": self._load(os.path.join(self.c_vi_dir, name)),
            "label1": "IR", "label2": "VI",
        }


class MSRSTestSet(Dataset):
    def __init__(self, cfg, condition):
        assert condition in ("clean", "degraded")
        root = cfg["paths"]["data_dir"]
        clear = os.path.join(root, "test", "clear")
        base = clear if condition == "clean" else os.path.join(root, "test", "jibian+noise+motion")

        ir_n = set(os.listdir(os.path.join(base, "ir")))
        vi_n = set(os.listdir(os.path.join(base, "vi")))
        c_ir = set(os.listdir(os.path.join(clear, "ir")))
        c_vi = set(os.listdir(os.path.join(clear, "vi")))
        self.ids = sorted(ir_n & vi_n & c_ir & c_vi)
        self.ir_dir = os.path.join(base, "ir")
        self.vi_dir = os.path.join(base, "vi")

        expected = cfg["data"].get("expected_test_pairs")
        if expected is not None and len(self.ids) != expected:
            raise ValueError(f"test expected {expected} pairs; got {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        return IMG_TRANSFORM(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        name = self.ids[idx]
        return {
            "id": name,
            "src1": self._load(os.path.join(self.ir_dir, name)),
            "src2": self._load(os.path.join(self.vi_dir, name)),
            "label1": "IR", "label2": "VI",
        }
