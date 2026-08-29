# Stage 2.6 — Area captioning

**Script:** `scripts/scene_area_classify.py` · **Module:** `services/vlm.py` (`VLMClient`).
**Input:** the area/object `SceneGraph` (`scene_graph.json`).
**Output:** the same graph with each `area` node's `name` + `desc` filled in.

## Mechanism

1. **Aggregate members.** For each `area` node, tally its member objects' label `votes` into a
   Counter → a compact summary string (e.g. `chair x4, table x2, monitor x1`).
2. **LLM caption.** The summary is sent to a `VLMClient` (`--model`, default
   `ollama:qwen2.5vl:3b`) under a structured Pydantic schema `AreaCaption(name, desc)`: a concise
   2–4 word `name` and a one-sentence `desc` of what kind of space it is.
3. **Write back.** `area.name`, `area.desc` are updated in place in the `SceneGraph`.

## Design rationale

- **No fixed room-type vocabulary** — the model infers a free-form name from the object
  distribution, so the same code works across building types without a label taxonomy.
- Captioning consumes only the **aggregated label distribution**, not imagery or geometry, so it is
  cheap and depends only on stage 2.5's membership — it can be re-run after any re-cluster.
- It writes into the same `scene_graph.json` (name/desc are core `SceneNode` fields), so downstream
  consumers see a single artifact whether or not captioning ran.

## Knobs

`--model` (any `VLMClient` backend: Ollama or OpenAI) · `--max-areas` (cap for testing; 0 = all).

## Notes

Optional stage — the areas from 2.5 are fully usable without captions. Needs a reachable VLM
backend; a stub encoder/validation run exercises the plumbing without one.
