"""Assignment with capacities, read from the job-machine edges that carry flow.

Jobs must go on machines, each job may use only some machines, and each machine
can take a limited number of jobs. Getting as many jobs placed as possible is a
maximum flow:

* source -> job, capacity 1 (or the job's demand),
* job -> each allowed machine, capacity 1,
* machine -> sink, capacity the machine's capacity.

A unit of flow from source to sink is one job placed on one machine. Capacities
are integers, so the flow is integral, every job-machine edge carries 0 or 1, and
the edge capacities are exactly the rules: a job takes no more than its demand,
uses only allowed machines, and no machine goes over capacity. Every feasible
placement is a flow and every integral flow is a feasible placement, so a maximum
flow places the most jobs possible.

The same network handles any problem of this shape. In project staffing, people
are the jobs, projects are the machines and each project's head count is its
capacity; see the example in :func:`schedule`.

How it reads the residual graph
-------------------------------
The solver leaves its final residual graph in the network, and the assignment
is read from it: a job-machine edge is in use when it carries flow, meaning its
residual capacity has fallen below its original capacity
(:meth:`~src.flow_network_solver.core.network.Network.flow_on`). A job with no
used edge did not fit.

Reverse edges are what stop an early choice from blocking a job. With J1 allowed
on M1 and M2, J2 allowed only on M1, and one slot on each, a first path J1 -> M1
leaves J2 stuck. The residual path source, J2, M1, J1, M2, sink then crosses the
reverse of J1 -> M1, moving J1 to M2 so that J2 can have M1.

================ ============================ =========
Function         Time                         Space
================ ============================ =========
schedule         O(F (V + E))                 O(V + E)
================ ============================ =========

Here V = jobs + machines + 2, E is the number of network edges, and F is the
flow value, at most the sum of the demands. Each augmenting path takes one BFS.

Examples:
    >>> allowed = {"J1": ["M1"], "J2": ["M1", "M2"], "J3": ["M1"], "J4": ["M2"]}
    >>> assignment, unassigned = schedule(
    ...     ["J1", "J2", "J3", "J4"], ["M1", "M2"], allowed, {"M1": 2, "M2": 1}
    ... )
    >>> assignment, unassigned
    ({'J1': 'M1', 'J2': 'M1', 'J4': 'M2'}, ['J3'])

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network

__all__ = ["schedule"]

#: Node ids in the scheduling network; jobs and machines follow from 2.
_SOURCE, _SINK = 0, 1


def _positions(labels: Iterable[Hashable], kind: str) -> Dict[Hashable, int]:
    """Map each label to its index, rejecting repeated labels.

    Args:
        labels: The job or machine labels, in order.
        kind: ``"job"`` or ``"machine"``, used in the error message.

    Returns:
        A dict ``{label: index}`` in input order.

    Raises:
        ValueError: If a label appears twice.

    Time Complexity:
        O(k) for k labels.

    Space Complexity:
        O(k).

    Examples:
        >>> _positions(["J1", "J2"], "job")
        {'J1': 0, 'J2': 1}
        >>> _positions(["J1", "J1"], "job")
        Traceback (most recent call last):
            ...
        ValueError: repeated job 'J1'
    """
    positions: Dict[Hashable, int] = {}
    for label in labels:
        if label in positions:
            raise ValueError(f"repeated {kind} {label!r}")
        positions[label] = len(positions)
    return positions


def schedule(
    jobs: Iterable[Hashable],
    machines: Iterable[Hashable],
    allowed: Mapping[Hashable, Sequence[Hashable]],
    capacities: Mapping[Hashable, int],
    demands: Optional[Mapping[Hashable, int]] = None,
) -> Tuple[Dict[Hashable, Union[Hashable, List[Hashable]]], List[Hashable]]:
    """Assign as many jobs as possible to allowed machines within capacity.

    Builds the network from the module docstring, runs Edmonds-Karp, and reads
    the assignment from the final residual graph: a job holds a machine when the
    edge between them carries flow.

    With ``demands``, a job of demand d asks for d different machines, because
    each job-machine edge has capacity 1. The flow then maximizes the number of
    job-machine pairs, not the number of jobs served in full, so a job can end up
    with only part of its demand. Serving as many whole jobs as possible is not a
    flow problem, and this function does not attempt it.

    Args:
        jobs: The job labels, hashable and distinct.
        machines: The machine labels, hashable and distinct.
        allowed: Maps a job to the machines it may use. A repeated machine counts
            once, and a job missing from the dict may use none.
        capacities: Maps each machine to how many jobs it can take, a
            non-negative ``int``.
        demands: Maps a job to the number of machines it needs, a positive
            ``int``. A job missing from the dict needs 1. ``None`` means every
            job needs 1.

    Returns:
        ``(assignment, unassigned)``. Without ``demands``, ``assignment`` maps
        each placed job to its machine. With ``demands``, it maps each job that
        got at least one machine to the list of its machines. Either way jobs
        follow input order, and machines follow the order of ``allowed[job]``.
        ``unassigned`` lists, in input order, each job that got less than it
        needs. A job with part of its demand is therefore in both.

    Raises:
        TypeError: If a capacity or demand is not an ``int``.
        ValueError: If ``jobs`` or ``machines`` repeats a label, ``allowed`` or
            ``demands`` names an unknown job, ``allowed`` names an unknown
            machine, a machine has no capacity, a capacity is negative, or a
            demand is below 1.

    Time Complexity:
        O(F (V + E)), where F is the flow value, at most the sum of the demands:
        each augmenting path adds at least one unit and takes one BFS.

    Space Complexity:
        O(V + E).

    Examples:
        Project staffing: people are the jobs, projects are the machines, and
        head counts are the capacities. Three seats exist for four people, so one
        person stays free.

        >>> people = ["Ana", "Ben", "Cho", "Dev"]
        >>> projects = ["Web", "Data"]
        >>> skills = {"Ana": ["Web"], "Ben": ["Web", "Data"], "Cho": ["Data"], "Dev": ["Data"]}
        >>> head_count = {"Web": 2, "Data": 1}
        >>> schedule(people, projects, skills, head_count)
        ({'Ana': 'Web', 'Ben': 'Web', 'Cho': 'Data'}, ['Dev'])

        J1 starts on M1 and moves to M2 through a reverse edge, so J2 fits too.

        >>> schedule(["J1", "J2"], ["M1", "M2"], {"J1": ["M1", "M2"], "J2": ["M1"]},
        ...          {"M1": 1, "M2": 1})
        ({'J1': 'M2', 'J2': 'M1'}, [])

        With ``demands`` each value is a list. J2 asks for two machines but is
        allowed only one, so it appears in both results.

        >>> schedule(["J1", "J2"], ["M1", "M2"], {"J1": ["M1", "M2"], "J2": ["M1"]},
        ...          {"M1": 2, "M2": 1}, demands={"J1": 2, "J2": 2})
        ({'J1': ['M1', 'M2'], 'J2': ['M1']}, ['J2'])
        >>> schedule(["J1"], ["M1"], {"J1": ["M2"]}, {"M1": 1})
        Traceback (most recent call last):
            ...
        ValueError: unknown machine 'M2' allowed for job 'J1'
    """
    job_at = _positions(jobs, "job")
    machine_at = _positions(machines, "machine")
    for job in allowed:
        if job not in job_at:
            raise ValueError(f"unknown job {job!r} in allowed")
    for job in demands or ():
        if job not in job_at:
            raise ValueError(f"unknown job {job!r} in demands")
    missing = [machine for machine in machine_at if machine not in capacities]
    if missing:
        raise ValueError(f"no capacity given for machine {missing[0]!r}")
    needs = {job: 1 if demands is None else demands.get(job, 1) for job in job_at}
    for job, need in needs.items():
        if need < 1:
            raise ValueError(f"demand of job {job!r} must be at least 1, got {need}")

    first_job, first_machine = 2, 2 + len(job_at)
    net = Network(first_machine + len(machine_at))
    for job, i in job_at.items():
        net.add_edge(_SOURCE, first_job + i, needs[job])
    for machine, j in machine_at.items():
        net.add_edge(first_machine + j, _SINK, capacities[machine])
    placements = []
    for job, i in job_at.items():
        for machine in dict.fromkeys(allowed.get(job, ())):
            if machine not in machine_at:
                raise ValueError(f"unknown machine {machine!r} allowed for job {job!r}")
            edge = net.add_edge(first_job + i, first_machine + machine_at[machine], 1)
            placements.append((job, machine, edge))

    edmonds_karp(net, _SOURCE, _SINK)

    held: Dict[Hashable, List[Hashable]] = {job: [] for job in job_at}
    for job, machine, edge in placements:
        if net.flow_on(edge) > 0:
            held[job].append(machine)
    unassigned = [job for job in job_at if len(held[job]) < needs[job]]
    if demands is None:
        return {job: got[0] for job, got in held.items() if got}, unassigned
    return {job: got for job, got in held.items() if got}, unassigned
