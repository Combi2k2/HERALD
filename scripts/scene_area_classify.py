#!/usr/bin/env python
"""scene_area_classify: LLM names + describes each area node from its members' labels.

No fixed label set -- the model infers a free-form name and one-sentence description from the
area's aggregated object-label distribution. Updates area.name / area.desc in the SceneGraph.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True, help="area/object SceneGraph (.json)")
    ap.add_argument("--out", type=Path, required=True, help="output SceneGraph (.json)")
    ap.add_argument("--model", default="ollama:qwen2.5vl:3b")
    ap.add_argument("--max-areas", type=int, default=0, help="cap areas captioned (0 = all; for testing)")
    args = ap.parse_args()

    from pydantic import BaseModel, Field

    from herald.scene.common.graph import SceneGraph
    from services.vlm import VLMClient, text_block

    class AreaCaption(BaseModel):
        name: str = Field(description="concise 2-4 word name for this area")
        desc: str = Field(description="one sentence describing what kind of space this is")

    g = SceneGraph.from_json(args.graph)
    by_uid = {n.uid: n for n in g.nodes}
    areas = [n for n in g.nodes if n.level == "area"]
    if args.max_areas:
        areas = sorted(areas, key=lambda a: -len(a.children))[: args.max_areas]

    client = VLMClient(args.model)
    for a in areas:
        votes: Counter = Counter()
        for cuid in a.children:
            o = by_uid.get(cuid)
            if o is not None:
                votes.update(o.votes or {o.name: 1})
        summary = ", ".join(f"{k} x{v}" for k, v in votes.most_common()) or "no labelled objects"
        prompt = (f"An indoor area contains these detected objects: {summary}. "
                  "Give it a concise name and a one-sentence description of what kind of space it is. "
                  "Do not use a fixed category list; infer freely from the objects.")
        cap = client.invoke([text_block(prompt)], schema=AreaCaption)
        a.name, a.desc = cap.name, cap.desc
        print(f"area {a.uid} ({len(a.children)} objs): {a.name} -- {a.desc}", flush=True)

    g.to_json(args.out)
    print(f"captioned {len(areas)} areas -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
