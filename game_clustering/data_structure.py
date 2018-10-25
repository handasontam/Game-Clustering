import numpy as np

class DisjointSetForest(object):
    """Efficient union find implementation for constructing the linkage matrix

    Parameters
    ----------

    size : int
        The total size of the set of objects to
        track via the union find structure.

    Attributes
    ----------

    is_component : array of bool; shape (size, 1)
        Array specifying whether each element of the
        set is the root node, or identifier for
        a component.
        
    Reference
    ----------
    
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    It provides near-constant-time operations (bounded by the inverse Ackermann function) 
    to add new sets, to merge existing sets, and to determine whether elements are in the same set.
    """
    def __init__(self, N):
        self.parent = np.array(range(2*N-1))
        self.parent = self.parent.astype(int)
        self.next_label = N 
        self.size = np.hstack((np.ones(N),  # initial all are singleton
                                   np.zeros(N-1)))  # for cluster that are formed later

    def union(self, m, n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        # self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1
        return

    def fast_find(self, n):
        p = n
        while p != self.parent[p]:
            p = self.parent[p]

        c = n
        while c != p:
            old_parent = self.parent[c]
            self.parent[c] = p
            c = old_parent
        return p


class DisjointSetForest_basic(object):

    """Efficient union find implementation.

    Parameters
    ----------

    size : int
        The total size of the set of objects to
        track via the union find structure.

    Attributes
    ----------

    is_component : array of bool; shape (size, 1)
        Array specifying whether each element of the
        set is the root node, or identifier for
        a component.
        
    Reference
    ----------
    
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    It provides near-constant-time operations (bounded by the inverse Ackermann function) 
    to add new sets, to merge existing sets, and to determine whether elements are in the same set.
    """

    def __init__(self, size):
        self._data = np.zeros((size, 2), dtype=np.intp)  # first column is the parent id, second column is the rank
        self._data.T[0] = np.arange(size)  # initialize the parent as the data poit itself (singleton partition)
        self.is_component = np.ones(size, dtype=np.bool)

    def union(self, x, y):
        """Union together elements x and y"""
        x_root = self.find(x)
        y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

        return 0

    def find(self, x):
        """
        Find the root or identifier for the component that x is in

        Find(x) follows the chain of parent pointers from x up the tree 
        until it reaches a root element, whose parent is itself. 
        This root element is the representative member of the set to which x belongs, 
        and may be x itself.
        """
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    def components(self):
        """Return an array of all component roots/identifiers"""
        return self.is_component.nonzero()[0]


class HierarchicalSolution(object):
    def __init__(self, d):
        self.solution = d

    def get_value_by_alpha(self, alpha):
        alpha = round(alpha, 8)
        if not self.solution:
            # dicrionary is empty, return empty set
            return set()
        if alpha in self.solution:
            return self.solution[alpha]
        else:
            # solution is indexed by the lowerbound alpha
            # find the corresponding alpha lowerbound from the solution set
            try:
                return self.solution[max(k for k in self.solution if k <= alpha)]
            except ValueError:
                return set()

    def get_solution(self):
        raise NotImplementedError
    
    def remove_duplicate(self):
        # remove duplicate
        duplicate_key = []
        alphas = sorted(list(self.solution.keys()))
        for a_1, a_2 in zip(alphas[0:-1], alphas[1:]):
            if self.solution[a_1] == self.solution[a_2]:
                duplicate_key.append(a_2)
        for key in duplicate_key:
            del self.solution[key]

    def get_alpha_prime(self, G):
        try:
            return min(k for k in self.solution if self.get_value_by_alpha(k) != set(G))
        except ValueError:
            return np.inf
    
    def get_items(self):
        return self.solution.items()

    def get_dict(self):
        return self.solution

class HierarchichalPartitionSolution(HierarchicalSolution):
    def __init__(self, d):
        super().__init__(d)
        # dictionary: keys are alpha, values are partition (set of frozenset)
        # self.solution = d

    def union(self):
        raise NotImplementedError

    def add_solution(self, alpha, partition):
        '''
        partition: set of set
        '''
        self.solution[round(alpha, 8)] = self.get_value_by_alpha(round(alpha, 8)).union(frozenset([partition]))

    @property
    def alpha_set(self):
        return set(self.solution.keys())
    # def get_alpha_prime(self):
    #     try:
    #         return min(k for k in self.solution if self.get_value_by_alpha(self.solution, k) != set(self.G))
    #     except ValueError:
    #         return np.inf

class HierarchicalCommunitySolution(HierarchicalSolution):
    def __init__(self, d):
        super().__init__(d)
        # self.solution = d
    
    def add_solution(self, alpha, community):
        '''
        community: a set
        '''
        self.solution[round(alpha, 8)] = self.get_value_by_alpha(alpha).union(set(community))


