import torch
import torch.nn.functional as F


def get_sam_encoder_output(sam, image):

    resized = F.interpolate(image, size=(1024, 1024), mode="bilinear",
                            align_corners=False)
    normalized = resized / 255.0 if resized.max() > 1 else resized
    with torch.no_grad():
        embeddings = sam.image_encoder(normalized)
    return embeddings
