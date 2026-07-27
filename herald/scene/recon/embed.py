"""SigLIP image-crop embedder for class-agnostic object association.

The Fuser identifies objects by the appearance of their mask crops rather than
by any class label or tracker id: each SAM2 mask is cropped to its bounding box
and embedded here, then associated to an existing object by cosine similarity
(gated by 3D proximity). SigLIP's image embeddings encode *what kind of thing*
a crop is, so the same object seen across frames lands close in this space.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


class SiglipEmbedder:
    """Embed RGB crops into L2-normalized SigLIP image vectors.

    __call__ takes a list of HxWx3 uint8 arrays (or PIL images) and returns an
    (N, D) float32 array of unit-norm embeddings, batched through the model.

    Uses ``get_image_features(...).pooler_output`` (image-only — no text prompt
    needed); after L2-normalization it points the same direction as the
    joint-forward ``image_embeds``, so cosine similarities live in the same
    space as SigLIP text<->image scoring.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.batch_size = max(1, batch_size)
        self.dim = int(self.model.config.vision_config.hidden_size)

    @torch.no_grad()
    def __call__(self, crops) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, self.dim), np.float32)
        imgs = [
            c if isinstance(c, Image.Image) else Image.fromarray(np.asarray(c, np.uint8))
            for c in crops
        ]
        out = []
        for i in range(0, len(imgs), self.batch_size):
            inp = self.processor(images=imgs[i:i + self.batch_size], return_tensors="pt").to(self.device)
            feat = self.model.get_image_features(pixel_values=inp["pixel_values"]).pooler_output
            feat = torch.nn.functional.normalize(feat, dim=-1)
            out.append(feat.float().cpu().numpy())
        return np.concatenate(out, axis=0)
