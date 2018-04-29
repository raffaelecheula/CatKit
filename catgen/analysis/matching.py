import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso


def reactant_indices(R1, R2, P, broken_bond):
    """Match the indices of a pair of reactants from a
     product and broken bond.

    Parameters:
    -----------
    R1 : networkx MultiGraph
        Graph representing reactant 1
    R2 : networkx MultiGraph
        Graph representing reactant 2
    P : networkx MultiGraph
        Graph representing the product
    broken_bond : list (2,)
        Indices representing the edge of the product
        to be removed.

    Returns:
    --------
    pindex: ndarrays (n,)
        Indices of the product graph sorted by the order of
        the reactants indices.
    """
    GM = nx.algorithms.isomorphism.GraphMatcher
    em = iso.numerical_edge_match('bonds', 1)
    nm = iso.numerical_node_match('number', 1)

    Pgraph = P.copy()
    u, v = broken_bond
    Pgraph.graph.remove_edge(u, v)
    Rgraph = R1 + R2

    gm = GM(Pgraph.graph, Rgraph.graph, edge_match=em, node_match=nm)

    gm.is_isomorphic()

    pindex = np.empty(len(Pgraph), dtype=int)
    for k, v in gm.mapping.items():
        pindex[k] = v

    return pindex
