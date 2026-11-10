"""Checkpoint validation for the nine supplied legacy models."""
from pathlib import Path

import torch


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    state = checkpoint.get("model", checkpoint)
    required = ("state_dict_fusion2", "state_dict_unet", "state_dict_vae")
    for name in required:
        if not isinstance(state.get(name), dict) or not state[name]:
            raise ValueError(f"Checkpoint is missing {name}: {path}")
        for key, value in state[name].items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Expected a tensor at {name}.{key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint tensor: {name}.{key}")
    return checkpoint


def copy_lora_parameters(module, state, component):
    """Reject missing, unexpected, or incorrectly shaped adapter parameters."""
    parameters = {name: value for name, value in module.named_parameters()
                  if "lora" in name}
    missing = sorted(set(parameters) - set(state))
    unexpected = sorted(set(state) - set(parameters))
    mismatched = [name for name in parameters.keys() & state.keys()
                  if parameters[name].shape != state[name].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{component} LoRA mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatch={mismatched[:5]}. "
            "Use the supplied legacy architecture and matching adapter ranks.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def validate_lora_pairs(state):
    count = 0
    for name, value in state.items():
        if "lora_A." in name:
            partner = name.replace("lora_A.", "lora_B.")
            if partner not in state or value.shape[0] != state[partner].shape[1]:
                raise ValueError(f"Invalid LoRA pair: {name}")
            count += 1
        elif "lora_B." in name:
            partner = name.replace("lora_B.", "lora_A.")
            if partner not in state:
                raise ValueError(f"Missing LoRA A tensor for {name}")
    if count == 0:
        raise ValueError("No LoRA A/B pairs found")
    return count
