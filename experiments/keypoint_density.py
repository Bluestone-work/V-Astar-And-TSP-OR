"""Optional sub-goal density cap for the validated planner in Algorithm 2."""

from global_planner import extract_key_points, planned_grid_route


def capped_key_points(path, occupancy, *, max_segment, **geometry):
    """Compress a validated path while capping the span of shortcuts."""
    return extract_key_points(path, occupancy, max_segment=max_segment, **geometry)


def density_route(grid, start, goal, *, compress=True, max_segment=None, **geometry):
    """Plan and validate a route with the requested shortcut density."""
    return planned_grid_route(grid, start, goal, compress=compress,
                              max_segment=max_segment, **geometry)
