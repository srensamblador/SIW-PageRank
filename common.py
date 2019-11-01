import time

import numpy as np


class Graph:
    def __init__(self, edges, undirected=False):
        # Keep a set with every node to refer to them via index
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = sorted(list(nodes))
        n = len(nodes)
        self.nodes = nodes

        # Generates the adyacency matrix from the edge list
        matrix = np.zeros((n, n))
        for a, b in edges:
            i, j = nodes.index(a), nodes.index(b)
            matrix[j][i] = 1

        if undirected: # For undirected graphs
            matrix = matrix + matrix.T - np.diag(matrix.diagonal())  # This calculates the symmetrical matrix

        self.m = self.__normalize_matrix(matrix)

    def __normalize_matrix(self, m):
        """
            From an adyacency matrix of 1 and 0s normalizes it such every node's column total equals 1
        """
        sum_columns = np.sum(m, axis=0)
        n = len(sum_columns)
        for j, j_total in enumerate(sum_columns):
            if j_total == 0: # Sink node
                for i in range(n):
                    m[i][j] = 1/n  # 1/n for every edge in a sink node
            else:
                for i in range(n):
                    m[i][j] /= j_total # For a normal node each edge is worth 1/number of outgoing edges
        return m

    def quadratic_error(self, v1, v2):
        """
            Calculates mean squared error from two vectors
            See: https://en.wikipedia.org/wiki/Mean_squared_error
        """
        if len(v1) != len(v2):
            raise TypeError("Both vectors must have the same length")

        sum_error = 0
        for i in range(len(v1)):
            sum_error += (v1[i] - v2[i]) ** 2
        return sum_error / len(v1)

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

        error = 1
        while error > limit:  # Iterate while mean quadratic error is greater than limit
            prev_v = v
            v = damping * np.dot(self.m, v) + (1 - damping) / n  # See https://wikimedia.org/api/rest_v1/media/math/render/svg/9f853c33de82a94b16ff0ea7e7a7346620c0ea04
            error = self.quadratic_error(prev_v, v)

        scores = {node: score for (node, score) in
                  zip(self.nodes, v.tolist())}  # Returns dictionary with the score for each node
        return scores


class WeightedGraph(Graph):
    """
        In case edge weights need to be specified, like in sentence text ranking where a similarity value is used
    """
    def __init__(self, edges):
        nodes = set()
        for edge, _ in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = sorted(list(nodes))
        n = len(nodes)
        self.nodes = nodes

        m = np.zeros((n, n))
        for edge, value in edges:
            i, j = nodes.index(edge[0]), nodes.index(edge[1])
            m[j][i] = value  # Specified value instead of 1

        self.m = m
