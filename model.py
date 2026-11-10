import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTokenizer, CLIPTextModel

from ftscm import FTSCM
from sam_features import get_sam_encoder_output
from segment_anything import sam_model_registry
from checkpoint_utils import copy_lora_parameters


def initialize_vae(sd_dir, lora_rank=4, use_decoder_lora=False):

    vae = AutoencoderKL.from_pretrained(sd_dir, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_target_modules_decoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
              "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and "encoder" in n:
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif pattern in n and ("decoder" in n or "post_quant_conv" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))

    l_target_modules_encoder = sorted(set(l_target_modules_encoder))
    l_target_modules_decoder = sorted(set(l_target_modules_decoder))
    lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",
                                   target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    if use_decoder_lora:
        lora_conf_decoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",
                                       target_modules=l_target_modules_decoder)
        vae.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        vae.set_adapters(["default_encoder", "default_decoder"])
    return vae, l_target_modules_encoder + l_target_modules_decoder


def initialize_unet(sd_dir, lora_rank=4):

    unet = UNet2DConditionModel.from_pretrained(sd_dir, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules = []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in",
              "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n:
                l_target_modules.append(n.replace(".weight", ""))
                break

    l_target_modules = sorted(set(l_target_modules))
    lora_conf = LoraConfig(r=lora_rank, init_lora_weights="gaussian",
                           target_modules=l_target_modules)
    unet.add_adapter(lora_conf, adapter_name="default")
    return unet, l_target_modules


class LFSDFusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sd_dir = cfg["paths"]["sd_dir"]
        model_cfg = cfg["model"]


        self.bdc_variant = bool(model_cfg.get("bdc_variant", False))
        self.use_sam = bool(model_cfg.get("use_sam", True))
        self.use_sd = bool(model_cfg.get("use_sd", True))
        self.use_sam_entropy = bool(model_cfg.get("use_sam_entropy", True))
        self.use_sam_correlation = bool(model_cfg.get("use_sam_correlation", True))
        self.use_lora = bool(model_cfg.get("use_lora", True))
        self.use_cdim = bool(model_cfg.get("use_cdim", True))
        self.init_angle = model_cfg.get("init_angle", None)
        self.init_angles = model_cfg.get("init_angles", None)
        if self.bdc_variant:
            from BDeformConv_layer4 import HaarWaveletProcessing
            self.fusion2 = HaarWaveletProcessing(channel_in=3, channel_out=3)
        else:
            self.fusion2 = FTSCM(
                channel_in=3, channel_out=3, sam_channels=256,
                use_cdim=self.use_cdim, init_angle=self.init_angle,
                init_angles=self.init_angles)
        self.fusion2.requires_grad_(True)


        self.sam = sam_model_registry["vit_b"](checkpoint=cfg["paths"]["sam_ckpt"])
        self.sam.eval()
        self.sam.requires_grad_(False)


        self.tokenizer = CLIPTokenizer.from_pretrained(sd_dir, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_dir, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # VAE / UNet
        self.unet_no_finetune = bool(model_cfg.get("unet_no_finetune", False))
        if self.unet_no_finetune:

            self.unet = UNet2DConditionModel.from_pretrained(sd_dir, subfolder="unet")
            self.unet.requires_grad_(False)
            self.lora_unet_modules = []
        elif not self.use_lora:

            self.unet = UNet2DConditionModel.from_pretrained(sd_dir, subfolder="unet")
            self.unet.requires_grad_(True)
            self.lora_unet_modules = []
        else:
            self.unet, self.lora_unet_modules = initialize_unet(sd_dir, model_cfg["lora_rank_unet"])
        self.vae_full = bool(model_cfg.get("vae_full_finetune", False))
        if not self.use_lora:

            self.vae = AutoencoderKL.from_pretrained(sd_dir, subfolder="vae")
            self.vae.requires_grad_(False)
            for n, p in self.vae.named_parameters():
                if n.startswith(("encoder.", "quant_conv.")):
                    p.requires_grad_(True)
            self.lora_vae_modules = []
        elif self.vae_full:

            self.vae = AutoencoderKL.from_pretrained(sd_dir, subfolder="vae")
            self.vae.requires_grad_(True)
            self.vae.train()
            self.lora_vae_modules = []
        else:
            self.vae, self.lora_vae_modules = initialize_vae(
                sd_dir, model_cfg["lora_rank_vae"],
                use_decoder_lora=bool(model_cfg.get("lora_vae_decoder", False)))
            self.vae.requires_grad_(False)
            for n, p in self.vae.named_parameters():
                if "lora" in n:
                    p.requires_grad_(True)

        self.timesteps1 = torch.tensor([int(model_cfg["timestep"])]).long()  # t*=1

        self.sam_lambda = float(model_cfg.get("sam_lambda", 0.1))
        self.lora_rank_unet = model_cfg["lora_rank_unet"]
        self.lora_rank_vae = model_cfg["lora_rank_vae"]
        self.sd_fp16 = bool(model_cfg.get("sd_fp16", True))


        self.sam_feature_cache = {}

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        self.text_encoder.eval()
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def precompute_sam_features(self, dataset, device):

        from tqdm.auto import tqdm
        self.sam.eval()
        print(f"precomputing SAM features for {len(dataset)} pairs ...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="SAM features"):
                item = dataset[i]
                sid = item["id"]
                if sid in self.sam_feature_cache:
                    continue
                s1 = item["src1"].unsqueeze(0).to(device)
                s2 = item["src2"].unsqueeze(0).to(device)
                vis_seg = get_sam_encoder_output(self.sam, s1)[0].cpu()
                ir_seg = get_sam_encoder_output(self.sam, s2)[0].cpu()
                self.sam_feature_cache[sid] = (vis_seg, ir_seg)

    def build_sam_prior(self, c_t1, c_t2, ids=None, return_debug=False):
        _, _, h, w = c_t1.size()
        if ids is not None and all(i in self.sam_feature_cache for i in ids):
            vis_seg = torch.stack([self.sam_feature_cache[i][0] for i in ids]).to(c_t1.device)
            ir_seg = torch.stack([self.sam_feature_cache[i][1] for i in ids]).to(c_t1.device)
        else:
            self.sam.eval()
            with torch.no_grad():
                vis_seg = get_sam_encoder_output(self.sam, c_t1)
                ir_seg = get_sam_encoder_output(self.sam, c_t2)
        vis_seg_raw = vis_seg
        ir_seg_raw = ir_seg
        vis_seg = F.interpolate(vis_seg, size=(h, w), mode="bilinear", align_corners=False)
        ir_seg = F.interpolate(ir_seg, size=(h, w), mode="bilinear", align_corners=False)


        vis_entropy = -torch.sum(F.softmax(vis_seg, dim=1) * F.log_softmax(vis_seg, dim=1), dim=1, keepdim=True)
        ir_entropy = -torch.sum(F.softmax(ir_seg, dim=1) * F.log_softmax(ir_seg, dim=1), dim=1, keepdim=True)


        if self.use_sam_entropy:
            total_entropy = vis_entropy + ir_entropy + 1e-8
            vis_weight = vis_entropy / total_entropy
            ir_weight = ir_entropy / total_entropy
        else:
            vis_weight = torch.full_like(vis_entropy, 0.5)
            ir_weight = torch.full_like(ir_entropy, 0.5)


        sam_features = vis_weight * vis_seg + ir_weight * ir_seg


        correlation = None
        complement_weight = None
        if self.use_sam_correlation:
            correlation = F.cosine_similarity(vis_seg, ir_seg, dim=1)
            complement_weight = 1.0 - torch.abs(correlation)
            complement_weight = complement_weight.unsqueeze(1)
            sam_features = sam_features + self.sam_lambda * complement_weight * (vis_seg + ir_seg)

        if return_debug:
            debug = {
                "sam_A": vis_seg_raw.detach(),
                "sam_B": ir_seg_raw.detach(),
                "entropy_A": vis_entropy.detach(),
                "entropy_B": ir_entropy.detach(),
                "weight_A": vis_weight.detach(),
                "weight_B": ir_weight.detach(),
                "correlation": correlation.detach() if correlation is not None else None,
                "complement_weight": complement_weight.detach() if complement_weight is not None else None,
                "sam_features": sam_features.detach(),
            }
            return sam_features, debug
        return sam_features

    def run_ftscm(self, c_t1, c_t2, ids=None):
        if self.bdc_variant:
            return self.fusion2(c_t1, c_t2)
        sam_features = None
        if self.use_sam:
            sam_features = self.build_sam_prior(c_t1, c_t2, ids=ids)
        return self.fusion2(c_t1, c_t2, sam_features)

    def run_ftscm_debug(self, c_t1, c_t2, ids=None):

        if self.bdc_variant:
            out = self.fusion2(c_t1, c_t2)
            return out, {"variant": "bdc"}
        sam_features, sam_debug = (self.build_sam_prior(c_t1, c_t2, ids=ids, return_debug=True)
                                   if self.use_sam else (None, {"variant": "no_sam"}))
        out, sdadc_debug = self.fusion2(c_t1, c_t2, sam_features, return_debug=True)
        return out, {**sam_debug, **sdadc_debug}

    def forward(self, c_t1, c_t2, prompt=None, prompt_embeds=None, return_all=True, ids=None):


        c_t = self.run_ftscm(c_t1, c_t2, ids=ids)
        sam_features = self.build_sam_prior(c_t1, c_t2, ids=ids) if (self.use_sam and not self.bdc_variant) else None

        if self.use_sd and prompt_embeds is None:
            prompt_embeds = self.encode_prompt(prompt)
        pos_caption_enc = prompt_embeds

        autocast_ctx = torch.autocast("cuda", torch.float16) if (self.sd_fp16 and c_t1.is_cuda) else nullcontext()

        with autocast_ctx:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

            g2 = self.vae.decode((encoded_control / self.vae.config.scaling_factor).float()).sample.detach()
            if self.use_sd:
                model_pred = self.unet(encoded_control, self.timesteps1.to(c_t1.device),
                                       encoder_hidden_states=pos_caption_enc.to(torch.float32)).sample

                x_denoised = (encoded_control - model_pred).float()
            else:

                x_denoised = encoded_control.float()
        g3 = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if return_all:
            return {"g1": c_t, "g2": g2, "g3": g3, "sam_features": sam_features}
        return g3

    def trainable_parameters(self):

        params = list(filter(lambda p: p.requires_grad, self.fusion2.parameters()))
        if self.vae_full or not self.use_lora:
            params += [p for p in self.vae.parameters() if p.requires_grad]
        else:
            params += [p for n, p in self.vae.named_parameters() if "lora" in n and p.requires_grad]
        if self.use_sd:
            if self.use_lora:
                params += [p for n, p in self.unet.named_parameters() if "lora" in n and p.requires_grad]
            else:
                params += [p for p in self.unet.parameters() if p.requires_grad]
        return params

    def save_model(self, outf):

        sd = {}
        sd["state_dict_fusion2"] = {k: v.cpu() for k, v in self.fusion2.state_dict().items()}
        sd["lora_unet_modules"] = self.lora_unet_modules
        sd["lora_vae_modules"] = self.lora_vae_modules
        sd["lora_rank_unet"] = self.lora_rank_unet
        sd["lora_rank_vae"] = self.lora_rank_vae
        if self.use_lora:
            sd["state_dict_unet"] = {k: v.cpu() for k, v in self.unet.state_dict().items() if "lora" in k}
        else:
            sd["state_dict_unet"] = {k: v.cpu() for k, v in self.unet.state_dict().items()}
        if self.vae_full:
            sd["state_dict_vae"] = {k: v.cpu() for k, v in self.vae.state_dict().items()}
        elif not self.use_lora:
            sd["state_dict_vae"] = {k: v.cpu() for k, v in self.vae.state_dict().items()
                                    if k.startswith(("encoder.", "quant_conv."))}
        else:
            sd["state_dict_vae"] = {k: v.cpu() for k, v in self.vae.state_dict().items() if "lora" in k}
        if outf is not None:
            torch.save(sd, outf)
        return sd

    def load_ckpt_from_state_dict(self, sd):
        self.fusion2.load_state_dict(sd["state_dict_fusion2"], strict=True)
        if self.use_lora:
            copy_lora_parameters(self.unet, sd["state_dict_unet"], "U-Net")
        else:
            self.unet.load_state_dict(sd["state_dict_unet"], strict=True)
        if self.vae_full:
            self.vae.load_state_dict(sd["state_dict_vae"], strict=True)
        elif not self.use_lora:
            self.vae.load_state_dict(sd["state_dict_vae"], strict=False)
        else:
            copy_lora_parameters(self.vae, sd["state_dict_vae"], "VAE")
