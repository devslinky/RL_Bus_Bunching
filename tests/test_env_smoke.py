"""Regression / smoke tests for the EXISTING TTC Route 29 environments.

These tests are a *guard rail*. They do NOT test any of the new code you will
write under ``student_project/`` -- they simply prove that the environment the
simulator already knows how to build ("ttc_route_29_north" and
"ttc_route_29_south") still constructs and still has a sane shape.

Why keep them green?
--------------------
Your job in this project is to REGENERATE the numbers that feed these
environments (headway, spacing, travel time, demand, OD) from the raw data --
see ``student_project/docs/03_data_to_env_mapping.md``. While you iterate it is
easy to accidentally break the existing environment (a bad stop id, a missing
link, a negative demand rate). If any assertion below turns red, stop and fix
it before moving on: it means the environment no longer builds correctly.

What is checked (all via the real, verified public API of the simulator):
  * ``Blueprint("ttc_route_29_north")`` and ``Blueprint("ttc_route_29_south")``
    both construct without error.
  * The network is a simple linear corridor: >= 40 nodes and exactly
    ``nodes - 1`` links.
  * The route schema holds exactly ONE route.
  * That route has a non-empty visit sequence, a start terminal, an end
    terminal, a positive schedule (dispatching) headway, and an OD rate table
    whose every entry is >= 0.
  * The precomputed distance-from-terminal is monotonically non-decreasing as
    you walk the route from the start terminal to the end terminal.

How to run (from the repo root ``/home/jiahao/Documents/busoperation``)::

    pytest student_project/tests/test_env_smoke.py -v

``tests/conftest.py`` puts the repo root on ``sys.path`` so ``import setup...``
resolves. Heavy third-party deps (networkx / matplotlib) are guarded with
``importorskip`` so a bare environment SKIPS rather than ERRORS.
"""

import pytest

# The two existing TTC environments this guard protects.
TTC_ENV_NAMES = ["ttc_route_29_north", "ttc_route_29_south"]

# The corridor is expected to be a linear route with ~48 nodes; keep a loose
# lower bound so small data changes don't make this brittle.
MIN_NODES = 40


def _import_blueprint():
    """Import ``Blueprint``, skipping the whole test if a heavy dep is absent.

    ``setup.blueprint`` pulls in networkx (graph) and matplotlib (via
    ``setup.network``). On a machine without them we want a SKIP, not an ERROR.
    """
    pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")
    # Imported here (not at module top) so collection never hard-fails.
    from setup.blueprint import Blueprint
    return Blueprint


def _node_count(network):
    """Total nodes = terminal nodes + stop nodes (every node is one or other)."""
    return (len(network.terminal_node_geometry_info)
            + len(network.stop_node_geometry_info))


def _link_count(network):
    """Total links (directed edges) in the network."""
    return len(network.link_geometry_info)


def _single_route(blueprint):
    """Return (route_id, Route_Details) for the one route, asserting there is
    exactly one."""
    details_by_id = blueprint.route_schema.route_details_by_id
    assert len(details_by_id) == 1, (
        f"expected exactly one route, found {len(details_by_id)}: "
        f"{list(details_by_id)}"
    )
    route_id = next(iter(details_by_id))
    return route_id, details_by_id[route_id]


@pytest.fixture(params=TTC_ENV_NAMES)
def env_name(request):
    """Parametrized: every test below runs once for NORTH and once for SOUTH."""
    return request.param


@pytest.fixture
def blueprint(env_name):
    """Construct the Blueprint for the current env. Construction itself is part
    of what we are testing -- if it raises, the test fails here."""
    Blueprint = _import_blueprint()
    return Blueprint(env_name)


def test_blueprint_constructs(blueprint, env_name):
    """The environment object builds and exposes a network + route schema."""
    assert blueprint.env_name == env_name
    assert blueprint.network is not None
    assert blueprint.route_schema is not None


def test_network_is_linear_corridor(blueprint):
    """>= 40 nodes and links == nodes - 1 (a simple open path)."""
    n_nodes = _node_count(blueprint.network)
    n_links = _link_count(blueprint.network)
    assert n_nodes >= MIN_NODES, f"expected >= {MIN_NODES} nodes, got {n_nodes}"
    assert n_links == n_nodes - 1, (
        f"expected a linear corridor (links == nodes - 1); "
        f"got {n_links} links for {n_nodes} nodes"
    )


def test_exactly_one_route(blueprint):
    """The TTC schema defines a single route."""
    route_id, _route = _single_route(blueprint)
    assert isinstance(route_id, str) and route_id != ""


def test_route_has_visit_sequence_and_terminals(blueprint):
    """The route has a non-empty visit sequence and both terminals set."""
    _route_id, route = _single_route(blueprint)

    assert route.visit_seq_stops, "visit_seq_stops must be non-empty"
    assert len(route.visit_seq_stops) >= 1

    assert route.terminal_id, "route must have a start terminal_id"
    assert route.end_terminal_id, "route must have an end_terminal_id"
    assert route.terminal_id != route.end_terminal_id, (
        "start and end terminals should differ"
    )


def test_schedule_headway_positive(blueprint):
    """Dispatching (schedule) headway must be a positive number of seconds."""
    _route_id, route = _single_route(blueprint)
    assert route.schedule_headway > 0, (
        f"schedule_headway must be > 0, got {route.schedule_headway}"
    )


def test_od_rate_table_non_negative(blueprint):
    """Every OD demand rate (pax/sec) must be non-negative."""
    _route_id, route = _single_route(blueprint)
    od_table = route.od_rate_table
    assert od_table, "od_rate_table must be non-empty"
    for origin, dest_rates in od_table.items():
        for dest, rate in dest_rates.items():
            assert rate >= 0, (
                f"negative OD rate {rate} for {origin} -> {dest}"
            )


def test_node_distance_monotonic(blueprint):
    """Distance-from-terminal is non-decreasing along the route.

    The node walk is [start terminal] + visit_seq_stops + [end terminal];
    ``blueprint.route_node_distance`` stores cumulative distance for each, so it
    must never step backwards.
    """
    route_id, route = _single_route(blueprint)
    node_dist = blueprint.route_node_distance[route_id]

    node_seq = [route.terminal_id] + list(route.visit_seq_stops) + [route.end_terminal_id]

    # Every node on the walk must have a recorded distance.
    for node_id in node_seq:
        assert node_id in node_dist, f"missing distance for node {node_id}"

    distances = [node_dist[node_id] for node_id in node_seq]
    for prev, curr in zip(distances, distances[1:]):
        assert curr >= prev, (
            f"distance decreased along the route: {prev} -> {curr} "
            f"(sequence: {distances})"
        )


if __name__ == "__main__":
    # Allow ``python student_project/tests/test_env_smoke.py`` as a convenience.
    raise SystemExit(pytest.main([__file__, "-v"]))
