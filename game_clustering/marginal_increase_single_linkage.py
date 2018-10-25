# from functools import lru_cache
from src.data_structure import DisjointSetForest, DisjointSetForest_basic
from src.utils import contract_community, contracted_nodes, f_approx, _tree_to_labels
from itertools import chain
import networkx as nx
import numpy as np
from itertools import combinations

from scipy.cluster.hierarchy import fcluster
from sklearn import metrics

class MarginalIncreaseSingleLinkage():

    def __init__(self, G, beta, verbose):
        self.G = G
        self.beta = beta
        self.verbose = verbose
        self.DSF = DisjointSetForest(self.G.number_of_nodes())
        self.linkage_matrix = np.empty((self.G.number_of_nodes() - 1, 4))
        self.vertex_mapping = {x:i for i, x in enumerate(self.G)}

    def f_approx(self, community):
        if len(community) == 1:
            return 0
        community_bar = set(self.G) - set(community)
        internal_wgt = 0
        external_wgt = 0
        for u, v, attr in self.G.in_edges(nbunch=community, data=True):
            # if (u != v):
            if u in community:
                internal_wgt += attr['weight']
            else:
                external_wgt += attr['weight']
    #     print('score: ',(beta * internal_wgt - (1 - beta) * external_wgt)/(len(community)))
        cardinality = np.sum([self.DSF.size[self.DSF.fast_find(self.vertex_mapping[v])] for v in community])
        return (self.beta * internal_wgt - (1 - self.beta) * external_wgt)/(cardinality**2)

    def find_best_pair(self):
        # Given the graph and j, find C_j containing j by repeatedly 
        # adding a node i that maxmizes (w(V\C_j, C_j) + beta * sum(internal_edge)) / size(C_j), 
        # until it cannot be increases
        # print(self.G.edges)
        best_score = -np.inf
        for i, j in combinations(self.G, 2):
            if self.G.has_edge(i, j) or self.G.has_edge(j, i):
                score = self.f_approx(community=frozenset([i, j]))
                if score > best_score:
                    best_pair = (i, j)
                    best_score = score
        return best_pair, best_score

    def a_b_clustering_approx(self):
        i = 0
        new_node_id = self.G.number_of_nodes()
        threshold = 0
        while self.G.number_of_nodes() != 1:
            # loop until G is contracted to only 1 vertex
            # get score and communities for all j:
            (v1, v2), score = self.find_best_pair()
            best_community = [v1, v2]
            print(best_community)
            print(score)
            v1_root = self.DSF.fast_find(self.vertex_mapping[v1])
            v2_root = self.DSF.fast_find(self.vertex_mapping[v2])
            if v1_root == v2_root:
                continue
            else:
                self.linkage_matrix[i][0] = v1_root
                self.linkage_matrix[i][1] = v2_root
                self.linkage_matrix[i][2] = -score  # negative in-cut
                # self.linkage_matrix[i][2] = threshold  # negative in-cut
                self.linkage_matrix[i][3] = self.DSF.size[v1_root] + self.DSF.size[v2_root]
                self.DSF.union(v1_root, v2_root)
                i += 1
            new_node_id = new_node_id + len(best_community) - 1
            self.G = contract_community(self.G, best_community, 1)
            threshold+=0.5
        # print(self.linkage_matrix)
        (a_b_pred,
                    probabilities_,
                    cluster_persistence_,
                    _condensed_tree,
                    linkage_matrix) = _tree_to_labels(self.linkage_matrix, min_cluster_size=2,
                                cluster_selection_method='eom', eom_degree=1)
        print(a_b_pred)
        linkage_matrix[:,2] = linkage_matrix[:,2]-min(linkage_matrix[:,2])
        from scipy.cluster.hierarchy import fcluster
        print(fcluster(linkage_matrix, 15, criterion='maxclust'))

        return linkage_matrix, a_b_pred

if __name__ == '__main__':

    G = nx.DiGraph()
    G.add_edges_from([(0,1,{'weight':0.1}), 
                    (1,2,{'weight':0.1}), 
                    (2,3,{'weight':0.1}), 
                    (3,4,{'weight':1})])
    num_v = G.number_of_nodes()
    beta = 1
    linkage_matrix = a_b_clustering_approx(G, beta, verbose=False)
    print(linkage_matrix)
    (a_b_pred,
                probabilities_,
                cluster_persistence_,
                _condensed_tree,
                linkage_matrix) = _tree_to_labels(linkage_matrix, min_cluster_size=2,
                            cluster_selection_method='eom', eom_degree=1)
    print(a_b_pred)