# from functools import lru_cache
from src.data_structure import DisjointSetForest
import networkx as nx
import numpy as np
from src.utils import f_approx, contract_community, _tree_to_labels
from itertools import chain
from joblib import Parallel, delayed

class MarginalIncreaseLinkage():
    def __init__(self, G, beta, n_jobs, verbose):
        self.G = G
        self.beta = beta
        self.verbose = verbose
        self.DSF = DisjointSetForest(self.G.number_of_nodes())
        self.linkage_matrix = np.empty((self.G.number_of_nodes() - 1, 4))
        self.vertex_mapping = {x:i for i, x in enumerate(self.G)}
        self.n_jobs = n_jobs

    def find_a_b_communitites_approx(self, j):
        # Given the graph and j, find C_j containing j by repeatedly 
        # adding a node i that maxmizes (w(V\C_j, C_j) + beta * sum(internal_edge)) / size(C_j), 
        # until it cannot be increases
        
        best_c_j = [j]  # start with singleton
        best_i_score = f_approx(G=self.G, community=frozenset(best_c_j), beta=self.beta, DSF=self.DSF, vertex_mapping=self.vertex_mapping)
        score_increased = True
        while score_increased:
            score_increased = False
            for node in best_c_j:
                for i in chain(self.G.successors(node), self.G.predecessors(node)):
                    if (i not in best_c_j):
                        # compute the score when add i to C_j
                        score = f_approx(G=self.G, community=frozenset(best_c_j + [i]), beta=self.beta, DSF=self.DSF, vertex_mapping=self.vertex_mapping)
                        if score > best_i_score:
                            best_i = i
                            best_i_score = score
                            if self.verbose:
                                print('found better community: ', best_c_j + [best_i])
                                print('best score: ', best_i_score)
                            score_increased = True
            if score_increased:
                best_c_j += [best_i]
        return best_c_j, best_i_score

    def a_b_clustering_approx(self):
        # contruct the linkage matrix
        threshold = 0.5
        i = 0
        new_node_id = self.G.number_of_nodes()
        while self.G.number_of_nodes() != 1:
            best_score = -np.inf
            # loop until G is contracted to only 1 vertex
            # get score and communities for all j:
            solutions = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(self.find_a_b_communitites_approx)(j) for j in self.G)
            solution = max(solutions, key=lambda x:x[1])
            best_community = solution[0]
            best_score = solution[1]
            # for j in self.G:
            #     c_j, score = self.find_a_b_communitites_approx(j)
                
            #     if score > best_score:
            #         best_community = c_j
            #         best_score = score
            print('best_community: ', best_community)
            print('best_score: ', best_score)
            for v1, v2 in zip(list(best_community)[:-1], list(best_community)[1:]):
                v1_root = self.DSF.fast_find(self.vertex_mapping[v1])
                v2_root = self.DSF.fast_find(self.vertex_mapping[v2])
                if v1_root == v2_root:
                    continue
                else:
                    self.linkage_matrix[i][0] = v1_root
                    self.linkage_matrix[i][1] = v2_root
                    self.linkage_matrix[i][2] = -best_score  # negative in-cut
                    self.linkage_matrix[i][3] = self.DSF.size[v1_root] + self.DSF.size[v2_root]
                    self.DSF.union(v1_root, v2_root)
                    i += 1
            new_node_id = new_node_id + len(best_community) - 1
            self.G = contract_community(self.G, best_community, 1)
    #         print(G.edges(data=True))
            threshold += 0.5
        (a_b_pred,
                    probabilities_,
                    cluster_persistence_,
                    _condensed_tree,
                    self.linkage_matrix) = _tree_to_labels(self.linkage_matrix, min_cluster_size=2,
                                cluster_selection_method='eom', eom_degree=1)
        self.linkage_matrix[:, 2] = self.linkage_matrix[:, 2] - min(self.linkage_matrix[:, 2])
        from scipy.cluster.hierarchy import fcluster
        print(fcluster(self.linkage_matrix, 15, criterion='maxclust'))
        return self.linkage_matrix, a_b_pred

# from scipy.cluster.hierarchy import fcluster
# from hdbscan._hdbscan_tree import (condense_tree,
#                             compute_stability,
#                             get_clusters,
#                             outlier_scores)
# from sklearn import metrics

# def _tree_to_labels(single_linkage_tree, min_cluster_size=10,
#                     cluster_selection_method='eom',
#                     allow_single_cluster=False,
#                     match_reference_implementation=False,
#                     eom_degree=1.0):
#     """Converts a pretrained tree and cluster size into a
#     set of labels and probabilities.
#     """
#     single_linkage_tree[:,2] = single_linkage_tree[:,2]**eom_degree
# #     print('single linkage tree: ', single_linkage_tree)
#     condensed_tree = condense_tree(single_linkage_tree,
#                                    min_cluster_size)
# #     print('condensed tree:', condensed_tree)
#     stability_dict = compute_stability(condensed_tree)
# #     print('stability dict:', stability_dict)
#     labels, probabilities, stabilities = get_clusters(condensed_tree,
#                                                       stability_dict,
#                                                       cluster_selection_method,
#                                                       allow_single_cluster,
#                                                       match_reference_implementation)

#     return (labels, probabilities, stabilities, condensed_tree,
#             single_linkage_tree)

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