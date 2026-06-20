from __future__ import annotations

import typing as tp
import torch
from collections import defaultdict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .serialization import GraphSpinSystem


def _get_node_signature(graph: GraphSpinSystem, node_idx: int, batch_idx: tuple) -> tuple:
    """
    Generate a hashable signature for a node based on its properties and local topology.
    Sorting components ensures the signature is invariant to internal principal value ordering.

    :param graph: The graph to extract the signature from.
    :param node_idx: The index of the node.
    :param batch_idx: The tuple index for the batch dimension (e.g., (0, 0) for 2D batch).
    :return: A tuple representing the node's structural and numerical signature.
    """
    n_type = graph.node_types[batch_idx + (node_idx,)].item()
    spin = round(graph.spins[batch_idx + (node_idx,)].item(), 4)
    comp = graph.components[batch_idx, node_idx, :]
    comp_sorted, _ = torch.sort(comp)
    comp_tuple = tuple(round(c.item(), 4) for c in comp_sorted)

    mask_src = graph.source == node_idx
    neighbors_src = graph.destination[mask_src].tolist()
    mask_dst = graph.destination == node_idx
    neighbors_dst = graph.source[mask_dst].tolist()
    neighbors = sorted(neighbors_src + neighbors_dst)

    neighbor_types = tuple(sorted([graph.node_types[batch_idx + (n,)].item() for n in neighbors]))
    return n_type, spin, comp_tuple, len(neighbors), neighbor_types


def _build_candidates(graph1: GraphSpinSystem, graph2: GraphSpinSystem, batch_idx: tuple, num_nodes: int) -> \
    tp.List[tp.List[int]]:
    """
    Build a list of candidate target nodes in graph2 for each source node in graph1
    based on matching signatures.

    :param graph1: The reference graph.
    :param graph2: The target graph to find candidates in.
    :param batch_idx: The tuple index for the batch dimension.
    :param num_nodes: Total number of nodes in the graphs.
    :return: A list where the i-th element contains a list of valid candidate indices in graph2 for node i in graph1.
    """
    sig_to_nodes = defaultdict(list)
    for j in range(num_nodes):
        sig_to_nodes[_get_node_signature(graph2, j, batch_idx)].append(j)

    return [sig_to_nodes[_get_node_signature(graph1, i, batch_idx)] for i in range(num_nodes)]


def _check_edge_constraints(graph1: GraphSpinSystem, graph2: GraphSpinSystem, node_i: int, node_j: int,
                            current_mapping: tp.List[int], edges_other: set) -> bool:
    """
    Check if mapping node_i (in graph1) to node_j (in graph2) violates edge constraints
    with already mapped nodes.

    :param graph1: The reference graph.
    :param graph2: The target graph.
    :param node_i: Node index in graph1.
    :param node_j: Proposed mapped node index in graph2.
    :param current_mapping: Current state of the mapping array (-1 if unmapped).
    :param edges_other: Set of (source, dest) tuples for graph2 for O(1) lookup.
    :return: True if the mapping is valid so far, False otherwise.
    """
    mask_src = graph1.source == node_i
    neighbors_src = graph1.destination[mask_src].tolist()
    mask_dst = graph1.destination == node_i
    neighbors_dst = graph1.source[mask_dst].tolist()

    for n in neighbors_src:
        mapped_n = current_mapping[n]
        if mapped_n != -1:
            if (node_j, mapped_n) not in edges_other:
                return False

    for n in neighbors_dst:
        mapped_n = current_mapping[n]
        if mapped_n != -1:
            if (mapped_n, node_j) not in edges_other:
                return False

    return True


def _validate_tensors(graph1: GraphSpinSystem, graph2: GraphSpinSystem, mapping: tp.List[int], rtol: float,
                      atol: float) -> bool:
    """
    Validate that components and angles match under the proposed mapping across all batch dimensions.

    :param graph1: The reference graph.
    :param graph2: The target graph.
    :param mapping: The complete node-to-node mapping from graph1 to graph2.
    :param rtol: Relative tolerance for tensor comparison.
    :param atol: Absolute tolerance for tensor comparison.
    :return: True if all tensors match within tolerance, False otherwise.
    """
    for i, j in enumerate(mapping):
        if not torch.allclose(graph1.components[..., i, :], graph2.components[..., j, :], rtol=rtol, atol=atol):
            return False
        if not torch.allclose(graph1.angles[..., i, :], graph2.angles[..., j, :], rtol=rtol, atol=atol):
            return False
    return True


def _backtrack_isomorphism(
        graph1: GraphSpinSystem,
        graph2: GraphSpinSystem,
        order: tp.List[int],
        idx: int,
        current_mapping: tp.List[int],
        used: tp.List[bool],
        candidates: tp.List[tp.List[int]],
        edges_other: set,
        rtol: float,
        atol: float
) -> bool:
    """
    Recursive backtracking search to find a valid graph isomorphism.

    :param graph1: The reference graph.
    :param graph2: The target graph.
    :param order: Order of nodes to process (most constrained first).
    :param idx: Current depth in the backtracking tree.
    :param current_mapping: Current state of the mapping array.
    :param used: Boolean array tracking which nodes in graph2 are already mapped.
    :param candidates: Precomputed candidate lists for each node in graph1.
    :param edges_other: Set of edges in graph2.
    :param rtol: Relative tolerance for tensor comparison.
    :param atol: Absolute tolerance for tensor comparison.
    :return: True if a valid isomorphism is found, False otherwise.
    """
    if idx == len(order):
        return _validate_tensors(graph1, graph2, current_mapping, rtol, atol)

    node_i = order[idx]
    for node_j in candidates[node_i]:
        if not used[node_j]:
            current_mapping[node_i] = node_j
            used[node_j] = True

            if _check_edge_constraints(graph1, graph2, node_i, node_j, current_mapping, edges_other):
                if _backtrack_isomorphism(graph1, graph2, order, idx + 1, current_mapping, used, candidates,
                                          edges_other, rtol, atol):
                    return True
            current_mapping[node_i] = -1
            used[node_j] = False

    return False


def are_graphs_equivalent(graph1: GraphSpinSystem, graph2: GraphSpinSystem, rtol: float = 1e-5,
                          atol: float = 1e-6) -> bool:
    """
    Function to check if two GraphSpinSystem instances are equivalent,
    allowing for arbitrary permutation of node ordering.

    :param graph1: The first graph to compare.
    :param graph2: The second graph to compare.
    :param rtol: Relative tolerance for tensor comparison.
    :param atol: Absolute tolerance for tensor comparison.
    :return: True if the graphs are structurally and numerically equivalent.
    """
    if not isinstance(graph2, type(graph1)):
        return False

    if graph1.components.shape != graph2.components.shape:
        return False

    batch_shape = graph1.components.shape[:-2]
    num_nodes = graph1.components.shape[-2]

    if num_nodes != graph2.components.shape[-2]:
        return False

    batch_idx = (0,) * len(batch_shape) if len(batch_shape) > 0 else ()

    edges_other = set(zip(graph2.source.tolist(), graph2.destination.tolist()))
    candidates = _build_candidates(graph1, graph2, batch_idx, num_nodes)
    if any(len(c) == 0 for c in candidates):
        return False
    order = sorted(range(num_nodes), key=lambda i: len(candidates[i]))

    mapping = [-1] * num_nodes
    used = [False] * num_nodes
    return _backtrack_isomorphism(
        graph1, graph2, order, 0, mapping, used, candidates, edges_other, rtol, atol
    )
