import hashlib
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from runtime import dataset_input_fingerprint

TRIPTYCH_QUESTION = (
    "<image>\nUSER: The left and the middle images are degraded, and the right image is their "
    "initial fused result. Describe the anatomical structures, tissues shown in the images in detail.\nASSISTANT:"
)

TRIPTYCH_QUESTION_SCENE = (
    "<image>\nUSER: The left image is a degraded IR image, the middle image is a degraded VI image, "
    "and the right image is their initial fused result. Describe the salient objects, thermal "
    "targets, textures and scene details shown in the images.\nASSISTANT:"
)


def load_llava(llava_dir, load_in_8bit=False, device=None):

    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if load_in_8bit and device.type != "cuda":
        raise ValueError("8-bit LLaVA loading requires a CUDA device")
    processor = AutoProcessor.from_pretrained(llava_dir)
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            llava_dir, quantization_config=quant_cfg, device_map={"": str(device)})
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            llava_dir, torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
            device_map={"": str(device)})
    model.eval()
    return model, processor


def tensor_to_pil(t):
    """Convert a tensor in [-1, 1] to a PIL image."""
    t = (t.detach().float().cpu().clamp(-1, 1) + 1) / 2
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def build_triptych(src1, src2, f_initial):

    imgs = [tensor_to_pil(src1), tensor_to_pil(src2), tensor_to_pil(f_initial)]
    h = imgs[0].height
    canvas = Image.new("RGB", (h * 3, h))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * h, 0))
    return canvas


def describe(processor, model, triptych_pil, label1, label2, domain="medical"):

    question = TRIPTYCH_QUESTION_SCENE if domain == "scene" else TRIPTYCH_QUESTION
    prompt = question.format(label1=label1, label2=label2)
    inputs = processor(text=prompt, images=triptych_pil, return_tensors="pt").to(model.device, model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(output[0], skip_special_tokens=True)


def _valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_triptychs_for_dataset(model, dataset, out_dir, device):

    from tqdm.auto import tqdm
    os.makedirs(out_dir, exist_ok=True)
    cur_hash = ftscm_snapshot_hash(model)
    input_hash = dataset_input_fingerprint(dataset)
    meta_path = os.path.join(out_dir, "ftscm_hash.json")
    meta = None
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except Exception:
            meta = None

    ids_labels = [(dataset[i]["id"], dataset[i]["label1"], dataset[i]["label2"])
                  for i in range(len(dataset))]

    have = {sid for sid in os.listdir(out_dir)
            if os.path.splitext(sid)[-1].lower() in (".png", ".jpg", ".jpeg")}
    valid = {sid for sid in have if _valid_image(os.path.join(out_dir, sid))}
    if (meta is not None and meta.get("hash") == cur_hash
            and meta.get("inputs_sha256") == input_hash
            and meta.get("labels") == {sid: [a, b] for sid, a, b in ids_labels}):
        print(f"triptych cache: validated model/input provenance; reusing {len(valid)} files")
    else:
        print("triptych cache: missing or stale provenance; rebuilding")
        valid = set()

    model.eval()
    paths = {}
    labels = {}
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="building triptychs"):
            sid, l1, l2 = ids_labels[i]
            path = os.path.join(out_dir, sid)
            if sid in valid:
                paths[sid] = (path, l1, l2)
                labels[sid] = [l1, l2]
                continue
            item = dataset[i]
            src1 = item["src1"].unsqueeze(0).to(device)
            src2 = item["src2"].unsqueeze(0).to(device)
            f_initial = model.run_ftscm(src1, src2, ids=[sid])
            triptych = build_triptych(item["src1"], item["src2"], f_initial[0].cpu())
            triptych.save(path)
            paths[sid] = (path, l1, l2)
            labels[sid] = [l1, l2]
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump({"hash": cur_hash, "inputs_sha256": input_hash, "labels": labels}, stream)
    return paths


def ftscm_snapshot_hash(model):

    h = hashlib.sha256()
    for k, v in model.fusion2.state_dict().items():
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class PromptCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.path = os.path.join(cache_dir, "prompt_cache.json")
        self.cache = {}
        self.meta = {}
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.cache = saved.get("captions", {})
            self.meta = saved.get("meta", {})

    def get(self, sample_id, default=None):
        return self.cache.get(sample_id, default)

    def update(self, captions, ftscm_hash, step, domain=None, dataset_hash=None):

        self.cache = dict(captions)
        self.meta = {"ftscm_hash": ftscm_hash, "step": step,
                     "num_captions": len(self.cache), "domain": domain,
                     "inputs_sha256": dataset_hash}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"captions": self.cache, "meta": self.meta}, f, ensure_ascii=False, indent=1)

    def is_stale(self, ftscm_hash, domain=None, dataset_hash=None):
        if self.meta.get("ftscm_hash") != ftscm_hash:
            return True
        if dataset_hash is not None and self.meta.get("inputs_sha256") != dataset_hash:
            return True
        return domain is not None and self.meta.get("domain") != domain
