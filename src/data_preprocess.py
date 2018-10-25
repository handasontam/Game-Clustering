import networkx as nx

def get_graph_from_data(data_path, directed, weighted):
    """
    Process the data into networkx DiGraph and run the algorithm
    :param data_path:
    :param directed:
    :return:
    """
    # process the data
    with open(data_path, 'r') as f:
        data = f.readlines()
        DG = nx.DiGraph()
        i=0
        for edges in data:
            u = int(edges.strip().split()[0])
            v = int(edges.strip().split()[1])
            if weighted:
                # the third column denotes the weight of the edge
                weight = float(edges.strip().split()[2])
            else:
                # set weight to be 1 for unweighted graph
                weight = 1
            if directed:
                DG.add_edges_from([(int(u), int(v), {'weight': weight})])
                if not DG.has_edge(int(v), int(u)):
                    DG.add_edges_from([(int(v), int(u), {'weight': 0})])
            else:
                # for undirected graph, we add the reverse path for each edge
                DG.add_edges_from([(int(u), int(v), {'weight': weight})])
                DG.add_edges_from([(int(v), int(u), {'weight': weight})])
            i+=1
    # print(len(list(DG.selfloop_edges())))
    # remove self loop
    # DG.remove_edges_from(DG.selfloop_edges())
    #print('original graph: ', G.edges(data=True))
    print('Graph loaded success')
    print('number of vertex: ', DG.number_of_nodes())
    print('number of edges: ', DG.number_of_edges())
    print()

    return DG