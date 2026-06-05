"""Small Z offsets for overlapping Rerun geometry."""


def layer_z_with_epsilon(base_z: float, entity_id: int | str) -> float:
    """Tiny per-entity Z offset so overlapping lines don't depth-fight."""
    key = int(entity_id) if isinstance(entity_id, int) else hash(entity_id)
    return base_z + (abs(key) % 997) * 0.04
