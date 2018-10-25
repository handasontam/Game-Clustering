import networkx as nx
import numpy as np
from itertools import chain
# from functools import lru_cache
# lru_cache is used for caching the results into memory, for the same input, we don't have to calculate again
import math
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed  # for parallel computing
import sys
from ctypes import *
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib1 = cdll.LoadLibrary(os.path.join(curr_path, 'ibfs_python3/libboost_python35.so.1.67.0'))
from .ibfs_python3 import ibfs_ext
from collections import defaultdict
from .data_structure import HierarchicalCommunitySolution, HierarchichalPartitionSolution


def intersection_point(m1, c1, m2, c2):
    """
    get the x, y coordinate of the intersection point of two line in 2D space
    :param m1: slope of line 1
    :param c1: y-intercept of line 1
    :param m2: slope of line 2
    :param c2: y-intercept of line 2
    :return: the x, y coordinate of the intersection point
    """
    # SHOULD CHECK DIVISION BY ZERO
    x = (c2 - c1) / (m1 - m2)
    y = (m1 * c2 - m2 * c1) / (m1 - m2)
    return x, y

class GameClustering(object):

    def __init__(self, G, beta, weight=None, ignore_nodes=set(), find_weaker_cluster=True, n_jobs=1, verbose=0):
        # Initialize

        if not weight:
            self.weight = 'weight'
            nx.set_edge_attributes(G, 1, self.weight)
        else:
            self.weight = weight
        nx.set_node_attributes(G=G, name='cluster_size', values=1)
        G = G.to_directed()
        for u, v, in G.copy().edges():
            if not G.has_edge(v, u):
                G.add_edges_from([(v, u, {'weight': 0})])
        self.in_degrees = dict(G.in_degree(nbunch=G.nodes, weight=weight))
        self.in_degrees_from_ignore_nodes = None
        self.N = G.number_of_nodes()
        self.M = G.number_of_edges()
        self.beta = beta
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.G = G
        self.find_weaker_cluster = find_weaker_cluster
        # sys.setrecursionlimit(max(10000, self.N*10))


    def get_vertex_id_mapping(self, ignore_nodes):
        vertices = set(self.G) - set(ignore_nodes)
        get_id_from_v = {v:x for x, v in enumerate(vertices)}
        return get_id_from_v

    def find_a_b_communitites_ibfs(self, j, alpha, ignore_nodes, verbose):
        all_ignore_nodes = [j] + list(ignore_nodes)
        get_id_from_v = self.get_vertex_id_mapping(all_ignore_nodes)
        # num_edges_between_ignore_nodes = self.G.subgraph
        num_edges_between_ignore_nodes = 0
        for edge in self.G.subgraph(set(self.G) - set(all_ignore_nodes)).edges():
            if (edge[0] != edge[1]):
                num_edges_between_ignore_nodes += 1
        # build j-augmentation graph
        j_aug = ibfs_ext.IBFSGraph()
        j_aug.initSize(self.N - len(all_ignore_nodes), num_edges_between_ignore_nodes)
        for i in (set(self.G) - set(all_ignore_nodes)):
            if self.G.has_edge(i, j):
                j_aug.addNode(get_id_from_v[i], alpha + self.in_degrees_from_ignore_nodes[i], self.G[i][j][self.weight] + self.beta * self.in_degrees[i])
            else:
                j_aug.addNode(get_id_from_v[i], alpha + self.in_degrees_from_ignore_nodes[i], self.beta * self.in_degrees[i])
        for edge in self.G.subgraph(set(self.G) - set(all_ignore_nodes)).edges(data=True):
            if (edge[0] != edge[1]):
                j_aug.addEdge(get_id_from_v[edge[0]], get_id_from_v[edge[1]], edge[2][self.weight], 0)
        j_aug.initGraph()
        # max-flow using IBFS
        mf = j_aug.computeMaxFlow()
        sink_label = [j]
        # add all vertex that is NOT on the source side (i.e. all vertex in the sink side)
        for v in (set(self.G) - set(all_ignore_nodes)):
            if not j_aug.isNodeOnSrcSide(get_id_from_v[v]):
                sink_label.append(v)
        return mf, frozenset(sink_label)

    def directed_cut_size(self, S, T=None):
        edges = nx.edge_boundary(self.G, S, T, data=self.weight, default=1)
        return sum(weight for u, v, weight in edges)


    def f_a_b_seperate(self, community):
        community_bar = frozenset(self.G) - frozenset(community)
        c = self.directed_cut_size(S=community_bar, T=community)
        b = np.sum([self.in_degrees[c] for c in community])
        a = np.sum([self.G.node[v]['cluster_size'] for v in community])
        return a, b, c


    def f_a_b(self, community, alpha):
        # the lower the better
        if len(community)==0:
            return 0
        else:
            a, b, c = self.f_a_b_seperate(community)
            return alpha * a - self.beta * b + c


    def find_alpha_ps(self, j, ignore_nodes, current_best_y_intercept, verbose, find_first_turning_pt_only=False):
        # variable that will be ammend in the recursion
        ps_t = dict()
        alpha_t = dict()
        f_0_t = dict()

        def _split_alpha(community_1, community_2, current_best_y_intercept={}, ignore_nodes=None, verbose=0):
            ''' fix beta recursively find all the split alpha in the Dilworth truncation '''
            # The y-intercept of community_1 and community_2
            f_0_c1 = self.f_a_b(community=community_1, alpha=0)
            f_0_c2 = self.f_a_b(community=community_2, alpha=0)

            # The slope of community_1 and community_2
            cardinality_c1 = len(community_1)
            cardinality_c2 = len(community_2)
            # The intersection point:
            alpha_bar, f_alpha_bar = intersection_point(m1=cardinality_c1,
                                                        c1=f_0_c1,
                                                        m2=cardinality_c2,
                                                        c2=f_0_c2)

            # communities of the intersection
            _, community = self.find_a_b_communitites_ibfs(j=j, alpha=alpha_bar, ignore_nodes=ignore_nodes,
                                                    verbose=verbose)
            f_alpha_intersection = self.f_a_b(community=community, alpha=alpha_bar)

            if verbose>=2:
                print('f_0_c1         ', f_0_c1)
                print('f_0_c2         ', f_0_c2)
                print('community_1    ')
                print(community_1)
                print('community_2')
                print(community_2)
                print('cardinality_c1: ', cardinality_c1)
                print('cardinality_c2: ', cardinality_c2)
                print('alpha_bar       ', alpha_bar)
                print('f_alpha_bar     ', f_alpha_bar)
                print('community of intersection: ')
                print(community)
                print('f_alpha of intersection: ', f_alpha_intersection)
                print('\n\n')

            # stopping criterion
            if (set(community_1) == set(community)) or (set(community_2) == set(community)) or (
                np.isclose(f_alpha_intersection, f_alpha_bar)) or (len(community) < cardinality_c1) or (
                current_best_y_intercept.get(len(community), np.inf) <= (f_alpha_intersection - alpha_bar * len(community))):
                
                ps_t[cardinality_c1] = community_1
                alpha_t[cardinality_c1] = alpha_bar
                f_0_t[cardinality_c1] = f_0_c1
                return
            elif f_alpha_intersection < f_alpha_bar:
                # pre-order recursion
                if not find_first_turning_pt_only:
                    # only split towards the trivial partition if we just want to get the compressive strength
                    _split_alpha(community_1, community, current_best_y_intercept, ignore_nodes, verbose)
                _split_alpha(community, community_2, current_best_y_intercept, ignore_nodes, verbose)
        _split_alpha(
                        community_1=frozenset({j}),  # singleton community only contains j
                        community_2=frozenset(frozenset(self.G.nodes) - frozenset(ignore_nodes)),  # trivial community contains all vertex
                        current_best_y_intercept=current_best_y_intercept,
                        ignore_nodes=ignore_nodes,
                        verbose=verbose)
        return ps_t, alpha_t, f_0_t


    def find_dominate_community(self, ignore_nodes, find_first_turning_pt_only=False):
        """
        find_dominate_community
        Parameters
        -----------
        G: a networkx DiGraph
        beta: float, [0,1]
            the beta value in alpha-beta communities
        weight: string, optional
            the attribute name of the weight, if not provided, we treat it as 
            an unweighted graph
        ignore_nodes: frozenset or set
            a set of integer representing the node we want to ignore,
            these nodes will be contracted to the source node when doing maxflow mincut
        verbose : int, default: 0
            control how much output message to show, can be 0, 1 or 2

        -----------
        Return
        best_ps : dict
            keys are integers, denoting the cardinality
            values are set denoting the set of vertices,
        alpha_solution : dict
            keys are integers, denoting the cardinality
            values are the alpha values,
        """
        if len(set(self.G.nodes)-set(ignore_nodes)) == 1:
            best_ps = {1: set([frozenset(set(self.G.nodes)-set(ignore_nodes))])}
            alpha_solution = {1: 0}
            return best_ps, alpha_solution
        # initialize
        best_ps = defaultdict(set)
        best_alpha = {}
        best_f_0 = {}
        ignore_nodes_in_degrees = dict(self.G.in_degree(nbunch=set(self.G.nodes)-set(ignore_nodes), weight=self.weight))
        self.ignore_nodes_out_neighbors = [out for n in ignore_nodes for out in self.G.successors(n) if out not in ignore_nodes]
        self.in_degrees_from_ignore_nodes = {j: np.sum([attr[self.weight] for nbr, _, attr in self.G.in_edges(nbunch=j, data=True) if nbr in ignore_nodes]) for j in self.G}
        smallest_in_degree_node = min(ignore_nodes_in_degrees, key=ignore_nodes_in_degrees.get)
        # print(self.ignore_nodes_in_degrees[smallest_in_degree_node])
        smallest_in_degree_nodes = {frozenset({k}) for k, v in ignore_nodes_in_degrees.items() if v == ignore_nodes_in_degrees[smallest_in_degree_node]}

        # find the y-intercept of the singleton partition with lowest mincut first
        # The trivial community
        best_ps[self.N - len(ignore_nodes)] = set([frozenset(self.G.nodes) - frozenset(ignore_nodes)])
        best_alpha[self.N - len(ignore_nodes)] = 0
        best_f_0[self.N - len(ignore_nodes)] = self.f_a_b(community=frozenset(self.G.nodes) - frozenset(ignore_nodes), alpha=0)
        # best singleton
        best_f_0[1] = self.f_a_b(community=frozenset({smallest_in_degree_node}), alpha=0)
        best_ps[1] = smallest_in_degree_nodes
        best_alpha[1] = -1
        #print('finding C_a_b_t for all t ...')
        solutions = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(self.find_alpha_ps)(j=j,
                                                        ignore_nodes=frozenset(ignore_nodes), 
                                                        current_best_y_intercept=best_f_0,
                                                        verbose= self.verbose, 
                                                        find_first_turning_pt_only=find_first_turning_pt_only) for j in (set(self.G) - set(ignore_nodes)))
        for C_a_b_t in solutions:
            ps_t, alpha_t, f_0_t = C_a_b_t
            for slope_t in ps_t.keys():
                if f_0_t[slope_t] < best_f_0.get(slope_t, np.inf):
                    best_f_0[slope_t] = f_0_t[slope_t]
                    best_alpha[slope_t] = alpha_t[slope_t]
                    best_ps[slope_t] = set([ps_t[slope_t]])
                elif f_0_t[slope_t] == best_f_0.get(slope_t, np.inf):
                    best_ps[slope_t] = best_ps[slope_t].union([ps_t[slope_t]])
                else:
                    continue
            
        
        ### filter the solution (getting the lowest picewise linear curve) ###
        alpha_solution = {}
        # add the trivial solution
        # alpha_solution[self.N - len(self.ignore_nodes)] = self.best_alpha[self.N - len(self.ignore_nodes)]
        previous_slope = self.N - len(ignore_nodes) + 1
        previous_intercept = best_f_0[self.N - len(ignore_nodes)]
        # sorted by the slope
        while True:
            # find the line with lowest intersection to the previous line
            lowest_intersection = np.inf
            temp_slope = np.inf  # when multiple lines intersect on the same point, choose the one with smallest slope
            for m, c in best_f_0.items():
                if m < previous_slope:
                    intersect_alpha, intersect_y = intersection_point(previous_slope, previous_intercept, m, c)
                    if np.isclose(intersect_y, lowest_intersection):
                        # when multiple lines intersect on the same point, choose the one with smallest slope
                        if m < temp_slope:
                            lowest_alpha = intersect_alpha
                            lowest_intersection = intersect_y
                            temp_slope = m
                    elif intersect_y < lowest_intersection:
                        lowest_alpha = intersect_alpha
                        lowest_intersection = intersect_y
                        temp_slope = m
            if temp_slope == np.inf:
                break
            alpha_solution[temp_slope] = lowest_alpha
            previous_slope = temp_slope
            previous_intercept = best_f_0[temp_slope]
        best_ps = {m:ps for m, ps in best_ps.items() if alpha_solution.get(m, None) is not None}
        # print({m:ps for m, ps in best_f_0.items() if alpha_solution.get(m, None) is not None})
        return best_ps, alpha_solution

    def fit(self):
        """
        Discover weaker communities recursively until there are only
        stop_n_node unlabled nodes
        -----------
        Return
        solutions : dict
            keys are alpha lower bound
            values are set of set: denoting the partition
        """
        # self.solutions = {}  # key: alpha lower bound, value: frozenset(frozenset(), ...), not include singleton
        self.solutions = HierarchichalPartitionSolution({})

        alpha_prime = -np.inf
        # self.ignore_nodes = defaultdict(set)
        self.ignore_nodes = HierarchicalCommunitySolution({})
        while alpha_prime != np.inf:
            B, alphas = self.find_dominate_community(ignore_nodes=self.ignore_nodes.get_value_by_alpha(alpha_prime))
            B_alpha = {round(alphas[cardinality], 8): B[cardinality] for cardinality in B.keys()}
            B_alpha = HierarchichalPartitionSolution(B_alpha)
            # print(B_alpha)
            alphas_to_consider = set(alphas.values()).union(self.solutions.alpha_set)
            alphas_to_consider = {a for a in alphas_to_consider if a >= 0}  # only consider alpha >= 0
            for alpha in sorted(alphas_to_consider, reverse=True):
                for c in B_alpha.get_value_by_alpha(alpha):
                    if not c.issubset(self.ignore_nodes.get_value_by_alpha(alpha)):
                        self.solutions.add_solution(alpha, c)
                    self.ignore_nodes.add_solution(alpha, c)
            alpha_prime = self.ignore_nodes.get_alpha_prime(self.G)
            if not self.find_weaker_cluster:
                break
        
        self.solutions.remove_duplicate()
        self.solutions = self.solutions.get_dict()

    def get_strength(self):
        """
        Discover weaker communities recursively until there are only
        stop_n_node unlabled nodes
        -----------
        Return
        solutions : dict
            keys are alpha lower bound
            values are set of set: denoting the partition
        """
        # self.solutions = {}  # key: alpha lower bound, value: frozenset(frozenset(), ...), not include singleton
        self.solutions = HierarchichalPartitionSolution({})

        alpha_prime = -np.inf
        # self.ignore_nodes = defaultdict(set)
        self.ignore_nodes = HierarchicalCommunitySolution({})
        B, alphas = self.find_dominate_community(ignore_nodes=self.ignore_nodes.get_value_by_alpha(alpha_prime), find_first_turning_pt_only=True)
        B_alpha = {round(alphas[cardinality], 8): B[cardinality] for cardinality in B.keys()}
        B_alpha = HierarchichalPartitionSolution(B_alpha)
        # print(B_alpha)
        alphas_to_consider = set(alphas.values()).union(self.solutions.alpha_set)
        alphas_to_consider = {a for a in alphas_to_consider if a >= 0}  # only consider alpha >= 0
        for alpha in sorted(alphas_to_consider, reverse=True):
            for c in B_alpha.get_value_by_alpha(alpha):
                if not c.issubset(self.ignore_nodes.get_value_by_alpha(alpha)):
                    self.solutions.add_solution(alpha, c)
                self.ignore_nodes.add_solution(alpha, c)
        alpha_prime = self.ignore_nodes.get_alpha_prime(self.G)

        self.solutions.remove_duplicate()
        if len(self.solutions.get_dict()) == 1:  # all the partition are all singleton (singleton dominate the trival)
            return None            
        alpha = sorted(list(self.solutions.alpha_set))[1]  # second smallest alpha

        return alpha
        
