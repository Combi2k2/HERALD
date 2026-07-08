"""RGB frames -> camera intrinsics, poses, and depth via VGGT-Omega."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

def load_model(checkpoint_path: str, device: str) -> VGGTOmega:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass device='cpu' to run on CPU.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return model.to(device)

def vggt_infer(
    image_paths: Sequence[str],
    image_resolution: int,
    model: VGGTOmega,
) -> dict[str, np.ndarray]:
    """Run VGGT-Omega over an RGB frame sequence.

    Returns {"K": (S,3,3), "c2w": (S,4,4), "depth": (S,H,W), "conf": (S,H,W)}.
    H/W are the model's working resolution, not the input resolution; K is in
    those pixels and c2w is camera-to-world (the model's world-to-camera
    extrinsics are inverted here).
    """
    device = next(model.parameters()).device
    images = load_and_preprocess_images(
        [str(p) for p in image_paths], image_resolution=image_resolution
    ).to(device)

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    w2c = extrinsic.float().cpu().numpy()  # (S,3,4) world-to-camera
    w2c = w2c[0] if w2c.ndim == 4 else w2c
    K = intrinsic.float().cpu().numpy()
    K = K[0] if K.ndim == 4 else K
    depth = predictions["depth"].float().cpu().numpy()
    depth = depth[0] if depth.ndim == 5 else depth
    depth = depth[..., 0] if depth.ndim == 4 else depth
    conf_t = predictions.get("depth_conf")
    if conf_t is not None:
        conf = conf_t.float().cpu().numpy()
        conf = conf[0] if conf.ndim == 4 else conf
    else:
        conf = np.ones_like(depth)

    rt = w2c[:, :3, :3].transpose(0, 2, 1)  # R^T per frame
    c2w = np.tile(np.eye(4), (len(w2c), 1, 1))
    c2w[:, :3, :3] = rt
    c2w[:, :3, 3] = -np.einsum("sij,sj->si", rt, w2c[:, :3, 3])
    return {"K": K, "c2w": c2w, "depth": depth, "conf": conf}
