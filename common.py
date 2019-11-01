import numpy as np


def quadratic_error(v1, v2):
    """
    Calculates mean squared error from two vectors
    """
    if len(v1) != len(v2):
        raise TypeError("Both vectors must have the same length")

    sum_error = 0
    for i in range(len(v1)):
        sum_error += (v1[i] - v2[i])**2
    return sum_error/len(v1)


class Graph:
    """
    def __init__(self, edges, undirected = False):
        self.graph = {}

        # Keeps a set of node names (A, B, C...) to refer to refer to them via index
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = sorted(list(nodes))
        self.nodes = nodes

        # Dictionary where:
        # Key {Node X}: [List of nodes that have an incoming edge from X]
        graph = {}
        for edge in edges:
            for node in edge:
                if nodes.index(node) not in graph:
                    graph[nodes.index(node)] = []
            graph[nodes.index(edge[0])].append(nodes.index(edge[1]))

        n = len(self.nodes)
        # m is the matrix representation of the graph
        self.m = np.zeros((n, n))

        for i in graph:
            if len(graph[i]) == 0:  # Sink node
                for j in range(n):
                    self.m[j][i] = 1/n  # Fill with 1/n for every row in the sink node's column to dampen afterwards
            else:
                for node in graph[i]:
                    self.m[node][i] = 1/len(graph[i])   # Edge value is 1/number of outgoing edges from the node
        
        # If the graph has to be undirected (e.g. TextRank)a
        if undirected:
            self.m = self.m + self.m.T - np.diag(self.m.diagonal())
            """

    def __init__(self, edges, undirected=False):
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = sorted(list(nodes))
        n = len(nodes)
        self.nodes = nodes

        m = np.zeros((n,n))
        for a, b in edges:
            i, j = nodes.index(a), nodes.index(b)
            m[j][i] = 1

        if undirected:
            m = m + m.T - np.diag(m.diagonal())

        norm = np.sum(m, axis=0)
        m = np.divide(m, norm, where=norm != 0)
        self.m = m



    def page_rank(self, damping=0.85, limit=1.e-8):
        """
        Calculates page rank of a graph
        :param damping: See https://en.wikipedia.org/wiki/PageRank#Damping_factor
        :param limit: Stop condition for the algorithm
        :return: dictionary of the form {node: score}
        """
        n = len(self.nodes)
        v = np.ones(n)
        v /= n  # Initialize v with 1/number of nodes

        print(len(v))

        error = 1
        while error > limit:  # Iterate while mean quadratic error is greater than limit
            prev_v = v
            v = damping*np.dot(self.m, v) + (1-damping)/n # See https://wikimedia.org/api/rest_v1/media/math/render/svg/9f853c33de82a94b16ff0ea7e7a7346620c0ea04
            error = quadratic_error(prev_v, v)

        scores = {node: score for (node, score) in zip(self.nodes, v.tolist())}  # Returns dictionary with the score for each node
        print(scores)
        return scores

