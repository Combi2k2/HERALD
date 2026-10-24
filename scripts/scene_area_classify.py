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

    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    from herald.scene.common.graph import SceneGraph
    from services.vlm import VLMClient, text_block

    class AreaCaption(BaseModel):
        contains: str = Field(description="comma-separated list of every distinguishable "
                               "object type present, including minor/background ones -- do "
                               "not summarize or omit any")
        name: str = Field(description="concise 2-4 word name for this area")
        desc: str = Field(description="ONE short sentence (at most 20 words) describing what "
                           "kind of space this is. Do not repeat phrases or ideas.")

    g = SceneGraph.from_json(args.graph)
    by_uid = {n.uid: n for n in g.nodes}
    areas = [n for n in g.nodes if n.level == "area"]
    if args.max_areas:
        areas = sorted(areas, key=lambda a: -len(a.children))[: args.max_areas]

    client = VLMClient(args.model)
    n_failed = 0
    for a in areas:
        votes: Counter = Counter()
        for cuid in a.children:
            o = by_uid.get(cuid)
            if o is not None:
                votes.update(o.votes or {o.name: 1})
        summary = ", ".join(f"{k} x{v}" for k, v in votes.most_common()) or "no labelled objects"
        prompt = (f"An area (indoor or outdoor -- infer which from the objects) contains these "
                  f"detected objects: {summary}. Give it a concise name and a one-sentence "
                  "description of what kind of space it is. Do not use a fixed category list; "
                  "infer freely from the objects.")
        cap = None
        for attempt in range(3):                                 # small local models occasionally
            try:                                                  # loop into a degenerate repetition
                cap = client.invoke([HumanMessage(content=[text_block(prompt)])], schema=AreaCaption)
                break
            except ValueError as e:
                print(f"area {a.uid}: attempt {attempt + 1} failed ({e}), retrying" if attempt < 2
                      else f"area {a.uid}: giving up after 3 attempts, falling back to raw label list",
                      flush=True)
        if cap is None:
            n_failed += 1
            a.name, a.desc = "unnamed area", "unnamed area"
            a.attrs["contains"] = summary
            continue
        a.name, a.desc = cap.name, cap.desc
        a.attrs["contains"] = cap.contains
        print(f"area {a.uid} ({len(a.children)} objs): {a.name} -- {a.desc} "
              f"[contains: {cap.contains}]", flush=True)

    g.to_json(args.out)
    print(f"captioned {len(areas)} areas ({n_failed} fell back to raw label list) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
