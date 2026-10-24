"""Planner stage 2: one LLM call that orders a retrieved shortlist of areas into a route.

Given the query, the requester's position, the robot's current position, and the shortlist of
candidate areas herald.scene.plan.descent.descend() retained, a single LLM call returns an
ordered sequence of area ids -- the "visit these, in this order" plan. Handles requests that
don't name a place directly (e.g. "bring me a wheelchair and return it to me") by injecting the
requester's own position as a synthetic candidate the LLM can select into the sequence, since
that's not something the scene graph -- built from object detections -- has any node for.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, create_model

REQUESTER_ID = "REQUESTER"
DONE_ID = "DONE"
STOP_ID = "STOP"
MAX_SEQUENCE_LEN = 12


@dataclass
class OrderResult:
    sequence: list
    reasoning: str | None = None  # only populated when use_reasoning=True


def _route_plan_schema(valid_ids: set[str], *, use_bounded_vocab: bool = True,
                        use_reasoning: bool = False) -> type[BaseModel]:
    """Builds a FRESH RoutePlan schema per call. Two independent toggles (the 2x2 ablation
    planner-eval-01/result.md's Addendum 5 runs):

    use_bounded_vocab: when True, `sequence`'s element type is constrained to exactly this
    call's valid_ids via a dynamic Literal -- not just "a str, checked after the fact." This is
    constrained decoding, not post-hoc validation: with a Literal/enum field, the JSON-schema-
    constrained decoding both Ollama (llama.cpp grammar) and Google GenAI's structured-output
    mode already use enforces the enum at the TOKEN level, so the model cannot generate an id
    outside the candidate list in the first place. When False, the field is plain `str` and the
    caller must post-hoc filter (the original, pre-Addendum-3 behavior) -- kept as an option
    specifically so "vanilla vs. bounded-vocab" can be compared head to head, not just assumed.

    use_reasoning: when True, adds a `reasoning` field BEFORE `sequence` in the schema. Field
    order matters for autoregressive generation under structured decoding -- a field listed
    first is generated first, so the model produces its free-text reasoning BEFORE being forced
    to commit to the (possibly enum-constrained) final answer, a simple model-agnostic
    chain-of-thought pattern that works even though qwen2.5vl:3b has no native "thinking" mode
    (unlike e.g. QwQ/Qwen3/DeepSeek-R1). When False, no reasoning field at all -- matches every
    prior run in this experiment.

    Two structural anti-runaway measures, both always on (found via real traces, not
    hypothesized -- see planner-eval-01/result.md's Addendum 5): (1) `STOP_ID` is added to the
    sequence's own valid-value set as an explicit, first-class "nothing more to add" action --
    unlike REQUESTER_ID (a real navigational action: go to the person) or DONE_ID
    (pick_next_stop()'s own, separate single-choice terminator), the one-shot list schema
    previously had NO explicit stop affordance at all; ending the list was left entirely to the
    JSON-array-closing grammar's implicit "continue vs. close" sampling, and once the model
    started repeating a pattern, "continue" could dominate indefinitely (observed: a 6-id cycle
    repeated 382 times in one real trace). order_route_core() strips STOP_ID and everything
    after it from the returned sequence -- it is a control token, never an executable stop. (2)
    `max_length=MAX_SEQUENCE_LEN` on the list itself is a hard structural backup in case the
    model still doesn't choose STOP -- belt and suspenders, not a replacement for the explicit
    stop token.

    (Loses VLMClient's per-schema-type cache, since every call gets a distinct class object; an
    acceptable, unavoidable cost since the valid set differs per call, and schema binding itself
    is local/cheap, not an API call.)"""
    seq_ids = set(valid_ids) | {STOP_ID}
    id_type = Literal[tuple(sorted(seq_ids))] if use_bounded_vocab else str
    fields: dict = {}
    if use_reasoning:
        fields["reasoning"] = (str, Field(description=(
            "Think step by step BEFORE answering: which candidate places actually satisfy the "
            "request (and which are irrelevant distractors), and what order they must be "
            "visited in given any ordering requirements in the request. Write this out first.")))
    fields["sequence"] = (list[id_type], Field(max_length=MAX_SEQUENCE_LEN,
                          description="ordered list of place ids to visit, drawn ONLY from the "
                          f"candidate list below (plus '{REQUESTER_ID}' meaning 'return to the "
                          f"person who asked') -- may repeat an id if the same place must be "
                          f"visited twice (e.g. picked up from and later returned to). End the "
                          f"list with the special id '{STOP_ID}' as its LAST entry once every "
                          f"needed place has been listed -- do not add anything after "
                          f"'{STOP_ID}'."))
    return create_model("RoutePlan", **fields)


def _next_stop_schema(valid_ids: set[str]) -> type[BaseModel]:
    """Same constrained-decoding approach as _route_plan_schema(), for the single-id case."""
    id_type = Literal[tuple(sorted(valid_ids))]
    return create_model(
        "NextStop",
        next_id=(id_type, Field(description="the id of the SINGLE place to visit next, drawn "
                 "ONLY from the candidate list below.")),
    )


def real_contains(area, graph_by_uid: dict, *, top_n: int = 12) -> str:
    """The area's REAL object list, from actual graph child-object votes -- NOT the `contains`
    attr (an LLM-generated caption). Empirically, that caption omits the object's own label in
    81% of (GT object, true-area) pairs on Hospital (see planner-eval-01/result.md's addendum):
    a caption like "Living Room: couch, sofa, tv, bench..." can genuinely be missing "door" even
    though a door geometrically IS in that room. Both the retrieval embedding text and this
    module's LLM prompt need this, since a caption the target object never appears in gives
    neither the embedding nor the LLM any signal that the area is actually correct. Sorted by
    vote count (most frequent first), capped at top_n to bound prompt size."""
    votes: Counter = Counter()
    for cuid in area.children:
        o = graph_by_uid.get(cuid)
        if o is not None:
            votes.update(o.votes or {o.name: 1})
    return ", ".join(f"{k} ({v})" if v > 1 else k for k, v in votes.most_common(top_n))


def _area_line(n, graph_by_uid: dict | None = None) -> str:
    contains = n.attrs.get("contains") or ""
    real = real_contains(n, graph_by_uid) if graph_by_uid is not None else ""
    combined = ", ".join(x for x in (real, contains) if x)
    return f"[{n.uid}] {n.name}: {n.desc} (contains: {combined})"


def order_route_core(query: str, candidates: list, requester_pos, robot_pos, model: str, *,
                      requester_desc: str | None = None, robot_desc: str | None = None,
                      graph_by_uid: dict | None = None, use_bounded_vocab: bool = True,
                      use_reasoning: bool = False) -> OrderResult:
    """General form of order_route() with both ablation toggles exposed (see
    _route_plan_schema()'s docstring) -- returns the sequence AND, if use_reasoning, the raw
    reasoning trace the model produced before committing to its answer. `order_route()` below is
    the production-path wrapper (bounded_vocab=True, no reasoning) that every other caller in
    this codebase uses; this function exists so an eval harness can run all four combinations
    head to head (planner-eval-01/result.md's Addendum 5).

    requester_desc/robot_desc: an optional natural-language landmark description (e.g. "near
    the Nurses' Station") to use IN PLACE OF the raw [x, y, z] position in the prompt -- raw
    coordinates carry no meaning an LLM can reason over (it can't judge "near" vs. "far" from
    two float triples without doing arithmetic it isn't asked to do), so a caller that can
    resolve a landmark (see herald.scene.plan.pipeline.nearest_area_desc) should prefer this.
    None falls back to the raw-position phrasing for callers with no graph context available."""
    from langchain_core.messages import HumanMessage

    from services.vlm import VLMClient, text_block

    listing = "\n".join(_area_line(n, graph_by_uid) for n in candidates)
    requester_where = requester_desc if requester_desc else f"at position {list(requester_pos)}"
    robot_where = robot_desc if robot_desc else f"at position {list(robot_pos)}"
    prompt = (
        f"A robot must satisfy this request from a person: \"{query}\"\n\n"
        f"The person who asked is {requester_where}. "
        f"The robot is currently {robot_where}.\n\n"
        f"Candidate places (id, name, description, contents):\n{listing}\n\n"
        f"Return the ordered sequence of place ids the robot must visit to satisfy the "
        f"request, using ONLY the ids listed above, plus the special id \"{REQUESTER_ID}\" if "
        f"the robot must return to the person (e.g. to deliver a fetched item). Order matters: "
        f"an item must be picked up before it is delivered. Only include places the request "
        f"actually needs, not every candidate listed. Finish the list with \"{STOP_ID}\" as its "
        f"last entry -- do not repeat places or add anything after \"{STOP_ID}\"."
    )
    valid_ids = {str(n.uid) for n in candidates} | {REQUESTER_ID}
    if not valid_ids:
        return OrderResult(sequence=[])
    schema = _route_plan_schema(valid_ids, use_bounded_vocab=use_bounded_vocab,
                                 use_reasoning=use_reasoning)
    # max_output_tokens: a bare RoutePlan response is a short id list, never near 512 tokens
    # under normal generation; a reasoning trace needs real room to think, hence the much higher
    # cap when use_reasoning -- both only bound the worst-case cost of a degenerate/looping
    # generation (see services.vlm.VLMClient's docstring), turning a ~20-minute hang into a fast,
    # retryable failure instead, not a token budget the model is expected to normally approach.
    # The STOP_ID sentinel + max_length on the schema (see _route_plan_schema's docstring) are
    # the REAL anti-runaway fix; this token cap is still a backstop, not the primary defense.
    client = VLMClient(model, max_output_tokens=2048 if use_reasoning else 512)
    # With use_bounded_vocab, constrained decoding (schema's per-entry Literal) already
    # guarantees every returned id is in valid_ids | {STOP_ID}; without it (the "vanilla"
    # ablation arm), fall back to the original post-hoc filter, now also keeping STOP_ID so the
    # truncation step below still finds it.
    seq_valid_ids = valid_ids | {STOP_ID}
    for attempt in range(3):
        try:
            result = client.invoke([HumanMessage(content=[text_block(prompt)])], schema=schema)
            raw = result.sequence if use_bounded_vocab else [s for s in result.sequence if s in seq_valid_ids]
            # Truncate at the first STOP_ID: it's a control token meaning "nothing more to add,"
            # never an executable destination -- strip it and everything the model (wrongly)
            # generated after it.
            seq = raw[:raw.index(STOP_ID)] if STOP_ID in raw else raw
            if seq:
                reasoning = getattr(result, "reasoning", None)
                return OrderResult(sequence=[int(s) if s != REQUESTER_ID else s for s in seq],
                                    reasoning=reasoning)
        except ValueError as e:
            print(f"order_route: attempt {attempt + 1} failed ({e})"
                  + ("" if attempt < 2 else ", giving up -- returning first candidate only"),
                  flush=True)
    return OrderResult(sequence=[candidates[0].uid] if candidates else [])


def order_route(query: str, candidates: list, requester_pos, robot_pos, model: str, *,
                 requester_desc: str | None = None, robot_desc: str | None = None,
                 graph_by_uid: dict | None = None) -> list:
    """Production path (bounded_vocab=True, no reasoning -- the config validated as best in
    Addendum 3/4). candidates: SceneNode list (herald.scene.plan.descent.descend()'s output).
    Returns the ordered id sequence -- ints for real area uids, or the REQUESTER_ID string
    sentinel. See order_route_core() for the full ablation surface (reasoning trace, vanilla
    vocab)."""
    return order_route_core(query, candidates, requester_pos, robot_pos, model,
                             requester_desc=requester_desc, robot_desc=robot_desc,
                             graph_by_uid=graph_by_uid).sequence


def _slotted_schema(item_labels: list[str], valid_ids: set[str]) -> type[BaseModel]:
    """One REQUIRED field per named item, instead of a free-length list -- removes the "how many
    things to list, and when to stop" decision from the model entirely (Addendum 5's new
    dominant failure mode after the STOP-sentinel fix: the model either looped past a sane
    length or stopped after 1-2 items regardless of how many were actually needed). A slot
    cannot be omitted (every field is required) and there's no list to loop into, so this
    removes both the over-inclusion (runaway repetition) and under-inclusion (premature
    stopping) failure modes by construction, not by better prompting around the same schema
    shape. Also closer to the paper's own described mechanism (decompose the query into
    sub-goals, then resolve each independently) than one big undifferentiated list ever was."""
    id_type = Literal[tuple(sorted(valid_ids))]
    fields: dict = {}
    for i, label in enumerate(item_labels):
        fields[f"place_for_item_{i}"] = (id_type, Field(description=(
            f"the id of the candidate place that has a '{label}' (item {i + 1} of "
            f"{len(item_labels)} in the request)")))
    fields["return_to_requester"] = (bool, Field(description=(
        f"whether the robot must return to the person (id '{REQUESTER_ID}') after fetching "
        "these items, e.g. because the request says to bring something back")))
    return create_model("SlottedPlan", **fields)


def order_route_slotted(query: str, item_labels: list[str], candidates: list, requester_pos,
                         robot_pos, model: str, *, requester_desc: str | None = None,
                         robot_desc: str | None = None, graph_by_uid: dict | None = None) -> OrderResult:
    """Structured alternative to order_route_core(): `item_labels` (the named items the query
    asks for, e.g. from query decomposition or -- in this experiment's evaluation -- the
    benchmark's own `required_labels`) each get one required schema slot instead of relying on
    a free-length list. See _slotted_schema()'s docstring for why. Sequence order follows
    item_labels' order, then REQUESTER_ID last if return_to_requester (this benchmark's
    unordered-fetch templates don't need item-to-item ordering; a caller with real ordering
    constraints would need a richer schema, not yet built)."""
    from langchain_core.messages import HumanMessage

    from services.vlm import VLMClient, text_block

    listing = "\n".join(_area_line(n, graph_by_uid) for n in candidates)
    requester_where = requester_desc if requester_desc else f"at position {list(requester_pos)}"
    robot_where = robot_desc if robot_desc else f"at position {list(robot_pos)}"
    items_listing = "\n".join(f"  item {i + 1}: {lab}" for i, lab in enumerate(item_labels))
    prompt = (
        f"A robot must satisfy this request from a person: \"{query}\"\n\n"
        f"The person who asked is {requester_where}. "
        f"The robot is currently {robot_where}.\n\n"
        f"Candidate places (id, name, description, contents):\n{listing}\n\n"
        f"The request asks for these {len(item_labels)} item(s):\n{items_listing}\n\n"
        f"For EACH item above, pick the single candidate place most likely to have it."
    )
    valid_ids = {str(n.uid) for n in candidates} | {REQUESTER_ID}
    if not valid_ids or not item_labels:
        return OrderResult(sequence=[])
    schema = _slotted_schema(item_labels, valid_ids)
    client = VLMClient(model, max_output_tokens=512)
    for attempt in range(3):
        try:
            result = client.invoke([HumanMessage(content=[text_block(prompt)])], schema=schema)
            seq = [getattr(result, f"place_for_item_{i}") for i in range(len(item_labels))]
            if getattr(result, "return_to_requester", False):
                seq.append(REQUESTER_ID)
            return OrderResult(sequence=[int(s) if s != REQUESTER_ID else s for s in seq])
        except ValueError as e:
            print(f"order_route_slotted: attempt {attempt + 1} failed ({e})"
                  + ("" if attempt < 2 else ", giving up -- returning first candidate only"),
                  flush=True)
    return OrderResult(sequence=[candidates[0].uid] if candidates else [])


def pick_next_stop(query: str, progress_desc: str, candidates: list, requester_pos, robot_pos,
                    model: str, *, requester_desc: str | None = None, robot_desc: str | None = None,
                    graph_by_uid: dict | None = None, allow_done: bool = True) -> str:
    """One LLM call that decides the SINGLE next destination, given a running summary of what's
    already been resolved -- the step primitive for herald.scene.plan.pipeline.
    plan_route_iterative()'s "resolve one destination, re-augment, resolve the next" loop
    (planner-eval-01's user-directed alternative to order_route()'s one-shot full-sequence call).
    Returns a candidate id (str), REQUESTER_ID, or DONE_ID (only if allow_done).

    allow_done: pass False on the FIRST step of a plan (nothing visited yet) -- empirically, the
    small local model would pick DONE immediately on 6/8 fresh single-item queries in testing
    (planner-eval-01/result.md's addendum), which can never be a legitimate answer since none of
    this benchmark's query templates are satisfied by zero actions. Structurally excluding DONE
    from the first step's valid options is a principled fix for a real answer the schema should
    never have allowed, not a benchmark-specific hack."""
    from langchain_core.messages import HumanMessage

    from services.vlm import VLMClient, text_block

    listing = "\n".join(_area_line(n, graph_by_uid) for n in candidates)
    requester_where = requester_desc if requester_desc else f"at position {list(requester_pos)}"
    robot_where = robot_desc if robot_desc else f"at position {list(robot_pos)}"
    done_clause = (f", or \"{DONE_ID}\" if the request is already fully satisfied by the "
                   f"progress so far and nothing more needs to be visited" if allow_done else "")
    prompt = (
        f"A robot is working step-by-step to satisfy this request from a person: \"{query}\"\n\n"
        f"Progress so far: {progress_desc}\n\n"
        f"The person who asked is {requester_where}. "
        f"The robot is currently {robot_where}.\n\n"
        f"Candidate places (id, name, description, contents):\n{listing}\n\n"
        f"Decide the SINGLE next place the robot should visit, using ONLY the ids listed above, "
        f"plus the special id \"{REQUESTER_ID}\" if the robot should return to the person now "
        f"(e.g. to deliver a fetched item){done_clause}."
    )
    valid_ids = {str(n.uid) for n in candidates} | {REQUESTER_ID} | ({DONE_ID} if allow_done else set())
    if not valid_ids:
        return DONE_ID if allow_done else REQUESTER_ID
    schema = _next_stop_schema(valid_ids)
    # max_output_tokens=128: one id is a tiny response, well under this even generously -- this only
    # bounds worst-case degenerate-generation cost (see services.vlm.VLMClient's docstring).
    client = VLMClient(model, max_output_tokens=128)
    # Constrained decoding (schema's Literal) already guarantees the returned id is in
    # valid_ids -- no post-hoc membership check needed, unlike the old plain-str-field version.
    for attempt in range(3):
        try:
            result = client.invoke([HumanMessage(content=[text_block(prompt)])], schema=schema)
            return result.next_id
        except ValueError as e:
            print(f"pick_next_stop: attempt {attempt + 1} failed ({e})"
                  + ("" if attempt < 2 else f", giving up -- {'treating as DONE' if allow_done else 'returning first candidate'}"),
                  flush=True)
    if allow_done:
        return DONE_ID
    return str(candidates[0].uid) if candidates else REQUESTER_ID
