import networkx as nx
import numpy as np
from itertools import chain
from hdbscan._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores)


def contract_community(G, community, new_node_id):
    H = G.copy()
    community = sorted(community)
    for node in community[1:]:
        H = contracted_nodes(H, community[0], node, self_loops=True)
#         print(H.edges(data=True))
#     H=nx.relabel_nodes(H, mapping={community[0]: new_node_id})
    return H

def contracted_nodes(G, u, v, self_loops=True, inplace=False):
    if inplace:
        H = G
    else:
        H = G.copy()    
    # edge code uses G.edges(v) instead of G.adj[v] to handle multiedges
    if H.is_directed():
        in_edges = [(w if w != v else u, u, d)
                    for w, _, d in G.in_edges(v, data=True)
                    if self_loops or w != u]
        out_edges = ((u, w if w != v else u, d)
                     for _, w, d in G.out_edges(v, data=True)
                     if self_loops or w != u)
        new_edges = list(chain(in_edges, out_edges))
    else:
        new_edges = ((u, w if w != v else u, d)
                     for x, w, d in G.edges(v, data=True)
                     if self_loops or w != u)
    H.remove_node(v)
    for u, v, d in new_edges:
        if H.has_edge(u, v):
            H.add_edge(u, v, weight=H.get_edge_data(u, v)['weight']+d['weight'])
        else:
            H.add_edge(u, v, weight=d['weight'])
    return H

def f_approx(G, community, beta, DSF, vertex_mapping):
    if len(community) == 1:
        return 0
    community_bar = set(G) - set(community)
    internal_wgt = 0
    external_wgt = 0
    for u, v, attr in G.in_edges(nbunch=community, data=True):
        if (u != v):
            if u in community:
                internal_wgt += attr['weight']
            else:
                external_wgt += attr['weight']
#     print('score: ',(beta * internal_wgt - (1 - beta) * external_wgt)/(len(community)))
    cardinality = np.sum([DSF.size[DSF.fast_find(vertex_mapping[v])] for v in community])
    return (beta * internal_wgt - (1 - beta) * external_wgt)/(cardinality**2)



def _tree_to_labels(single_linkage_tree, min_cluster_size=10,
                    cluster_selection_method='eom',
                    allow_single_cluster=False,
                    match_reference_implementation=False,
                    eom_degree=1.0):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    single_linkage_tree[:,2] = single_linkage_tree[:,2]**eom_degree
#     print('single linkage tree: ', single_linkage_tree)
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
#     print('condensed tree:', condensed_tree)
    stability_dict = compute_stability(condensed_tree)
#     print('stability dict:', stability_dict)
    labels, probabilities, stabilities = get_clusters(condensed_tree,
                                                      stability_dict,
                                                      cluster_selection_method,
                                                      allow_single_cluster,
                                                      match_reference_implementation)

    return (labels, probabilities, stabilities, condensed_tree,
            single_linkage_tree)
