import argparse
import networkx as nx
from marginal_increase_clustering import MarginalIncreaseClustering
from data_preprocess import get_graph_from_data
import numpy as np
import os
import pickle


def run(core, data_path, label_path, directed, weighted, beta, output_path, ignore_nodes=[], verbose=1):
    """
    Process the data into networkx DiGraph and run the algorithm
    :param core:
    :param data_path:
    :param directed:
    :param beta:
    :return:
    """
    # process the data
    DG = get_graph_from_data(data_path=data_path, directed=directed, weighted=weighted)
    MIC = MarginalIncreaseClustering(G=DG, beta=beta, weight='weight', n_jobs=core, verbose=verbose)
    MIC.fit()
    for alpha, cluster in sorted(MIC.solutions.items()):
        print('alpha: ', alpha)
        print('cluster: ', cluster)
    return DG


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finding the alpha-beta dominant communities')
    parser.add_argument('--cpu', type=int, required=True, help='number of cpu core to run the algorithm in parallel')
    parser.add_argument('--data', type=str, required=True, help='the file path of the graph data')
    parser.add_argument('--output', type=str, required=True, help='the output directory')
    # boolean argument deciding whether the graph is directed/undirected
    parser.add_argument('--directed', dest='directed', action='store_true')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=True)
    # boolean argument deciding whether the graph is weighted/unweighted
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)
    
    parser.add_argument('--beta', type=float, required=True, help='the beta value to use, range: [0,1]')


    args = parser.parse_args()

    print(args)
    print(args.data)
    run(core=args.cpu, data_path=args.data, label_path=args.label, directed=args.directed, weighted=args.weighted, beta=args.beta, output_path=args.output)
    # best_partition(core=args.cpu, data_path=args.data, label_path=args.label, directed=args.directed, weighted=args.weighted, beta=args.beta)
