"""VGGT geometry backend — the model "brain" for depth + camera pose.

This is the only real :class:`~services.reconstruction.base.GeometryBackend`.
The wrapper deliberately mirrors R3's ergonomics (the code style the project
likes): a thin class over the model, lazy weight loading, and per-frame outputs
shaped like R3's ``infer.py`` (depth, confidence, and a ``camera`` = pose +
intrinsics per frame). It does **not** depend on R3 — only on VGGT.

torch and the ``vggt`` package are imported lazily inside methods so that this
module (and the wider HERALD package) imports cleanly in a torch-free env; the
heavy stack is only touched when :meth:`VGGTBackend.infer` actually runs.

VGGT emits all frames' poses in one forward pass in a shared, up-to-scale world
frame. For very long streams a sliding-window/online scheme (à la R3) would be
needed; M1 runs a single forward pass and relies on ``frame_stride`` /
``max_frames`` upstream to bound cost.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from services.reconstruction.base import FrameGeometry


def _to_4x4(extrinsic: np.ndarray) -> np.ndarray:
    """Promote a (3, 4) or (4, 4) extrinsic to a homogeneous (4, 4) matrix."""
    e = np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (4, 4):
        return e
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = e
    return out


def _invert_se3(mat: np.ndarray) -> np.ndarray:
    """Invert a rigid (4, 4) transform: world-from-camera <-> camera-from-world."""
    r = mat[:3, :3]
    t = mat[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


class VGGTBackend:
    """Run VGGT over a set of frames -> list[:class:`FrameGeometry`]."""

    model_id = "vggt-1b"

    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        *,
        device: str | None = None,
        image_size: int = 518,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self._requested_device = device
        self._model = None
        self._device = None
        self._dtype = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            from vggt.models.vggt import VGGT

            device = torch.device(
                self._requested_device
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            # bf16 on Ampere+, else fp16 on GPU, fp32 on CPU.
            if device.type == "cuda":
                cap = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                dtype = torch.float32
            model = VGGT.from_pretrained(self.model_name).to(device).eval()
            self._model, self._device, self._dtype = model, device, dtype
        return self._model, self._device, self._dtype

    def infer(self, frames: Sequence[object]) -> list[FrameGeometry]:
        paths = [str(f) for f in frames]
        if not paths:
            return []

        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model, device, dtype = self._ensure_model()
        images = load_and_preprocess_images(paths).to(device)  # (S, 3, H, W)
        H, W = int(images.shape[-2]), int(images.shape[-1])

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ):
                preds = model(images[None])  # add batch dim -> (1, S, 3, H, W)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
        extrinsic = extrinsic[0].float().cpu().numpy()  # (S, 3, 4) w2c, OpenCV
        intrinsic = intrinsic[0].float().cpu().numpy()  # (S, 3, 3)
        depth = preds["depth"][0].float().cpu().numpy()  # (S, H, W, 1) or (S, H, W)
        conf_t = preds.get("depth_conf")
        conf = conf_t[0].float().cpu().numpy() if conf_t is not None else None

        out: list[FrameGeometry] = []
        for i in range(extrinsic.shape[0]):
            c2w = _invert_se3(_to_4x4(extrinsic[i]))
            d = depth[i]
            d = d[..., 0] if d.ndim == 3 else d
            out.append(
                FrameGeometry(
                    index=i,
                    K=intrinsic[i],
                    c2w=c2w,
                    depth=d,
                    conf=conf[i] if conf is not None else None,
                    rgb_ref=paths[i],
                )
            )
        return out
