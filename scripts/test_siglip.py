"""Smoke-test a SigLIP / SigLIP2 multimodal embedding model.

Exercises the three things you actually use a SigLIP model for:
  1. zero-shot image<->text scoring (sigmoid, per-label — SigLIP's hallmark, NOT softmax),
  2. raw L2-normalized image / text embeddings (dims + norms),
  3. image<->image cosine similarity (same-scene vs cross-scene sanity check).

Runs on real TartanGround frames by default so the scores mean something for HERALD.

    uv run python scripts/test_siglip.py                       # defaults: Hospital frame, SigLIP2 base
    uv run python scripts/test_siglip.py --model google/siglip-so400m-patch14-384
    uv run python scripts/test_siglip.py --image path/to.png --labels "a car,a tree,a building"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_DATA = Path.home() / "HERALD/data/tartanground/Hospital/Data_omni/P0000/image_lcam_front"
_DEF_IMAGE = _DATA / "000000_lcam_front.png"
_DEF_IMAGE2 = _DATA / "000200_lcam_front.png"  # later frame, same trajectory
# TartanGround "Hospital/P0000" opens on an office room (desk, chair, cabinet), so the
# defaults include an accurate label plus clear distractors to show the score spread.
_DEF_LABELS = (
    "an office room with a desk and chair,an office chair,a cabinet full of binders,"
    "a bedroom,a kitchen,a forest,a construction site,a car on a street,a person"
)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"image not found: {path}")
    return Image.open(path).convert("RGB")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/siglip2-base-patch16-224")
    p.add_argument("--image", type=Path, default=_DEF_IMAGE)
    p.add_argument("--image2", type=Path, default=_DEF_IMAGE2, help="second image for image<->image similarity")
    p.add_argument("--labels", default=_DEF_LABELS, help="comma-separated candidate text prompts")
    p.add_argument("--template", default="This is a photo of {}.",
                   help="prompt template wrapped around each label (SigLIP's canonical form)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    from transformers import AutoModel, AutoProcessor

    print(f"loading {args.model} on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.perf_counter() - t0:.1f}s | {n_params / 1e6:.0f}M params")

    images = [_load_image(args.image)]
    if args.image2.exists():
        images.append(_load_image(args.image2))
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    prompts = [args.template.format(lbl) for lbl in labels]

    # One forward over BOTH images + all label prompts. SigLIP needs padding='max_length'
    # (fixed 64-token context), unlike CLIP. out.{image,text}_embeds are the projected,
    # ready-to-use embeddings (get_image_features returns a backbone object here, not them).
    inputs = processor(text=prompts, images=images, padding="max_length", return_tensors="pt").to(args.device)
    t0 = time.perf_counter()
    out = model(**inputs)
    dt = time.perf_counter() - t0

    # ---- 1. zero-shot image<->text (SigLIP uses SIGMOID: each label scored independently) ----
    # sigmoid(scale*cosine + bias): the large negative bias puts the ~50% boundary at a
    # cosine of -bias/scale, so scores look low unless a label truly matches. NOT a softmax
    # (doesn't sum to 1). We print cosine too so the calibration is legible.
    img_n = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    txt_n = torch.nn.functional.normalize(out.text_embeds, dim=-1)
    cosines = (img_n[0] @ txt_n.T).float().cpu().numpy()
    probs = torch.sigmoid(out.logits_per_image[0]).float().cpu().numpy()  # per-label match prob in [0,1]
    boundary = -model.logit_bias.item() / model.logit_scale.exp().item()
    print(f"\n[zero-shot] {args.image.name}  ({dt * 1e3:.0f} ms, {len(images)} imgs x {len(labels)} labels)")
    print(f"  (sigmoid 50% boundary at cosine ~{boundary:.3f})")
    print(f"  {'sigmoid':>7} {'cosine':>7}  label")
    for idx in np.argsort(-probs)[: args.topk]:
        print(f"  {probs[idx] * 100:6.1f}% {cosines[idx]:+7.3f}  {labels[idx]:<28} {'#' * int(probs[idx] * 40)}")

    # ---- 2. raw embeddings (projected, then L2-normalized) ----
    img_emb, txt_emb = out.image_embeds, out.text_embeds
    print(f"\n[embeddings] image {tuple(img_emb.shape)}  text {tuple(txt_emb.shape)}  "
          f"| dim {img_emb.shape[-1]} | img L2 pre-norm {img_emb[0].norm().item():.2f}")

    # ---- 3. image<->image cosine similarity (two frames of the same trajectory => high) ----
    if len(images) > 1:
        img_n = torch.nn.functional.normalize(img_emb, dim=-1)
        cos = float((img_n[0] * img_n[1]).sum().item())
        print(f"\n[image<->image] {args.image.name} vs {args.image2.name}: cosine {cos:+.3f}")
    else:
        print(f"\n[image<->image] skipped ({args.image2} not found)")

    print("\nOK — SigLIP model is functional.")


if __name__ == "__main__":
    main()
