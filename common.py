import numpy as np


class Graph:
    def __init__(self, edges):
        self.graph = {}

        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        nodes = sorted(list(nodes))
        self.nodes = nodes
        print(self.nodes)

        graph = {}
        for edge in edges:
            for node in edge:
                if nodes.index(node) not in graph:
                    graph[nodes.index(node)] = []
            graph[nodes.index(edge[0])].append(nodes.index(edge[1]))

        n = len(self.nodes)
        self.m = np.zeros((n, n))
        for i in graph:
            if len(graph[i]) == 0:
                for j in range(n):
                    self.m[j][i] = 1/n  # Fill with 1/n for every row in the sink node's column to dampen afterwards
            else:
                for node in graph[i]:
                    self.m[node][i] = 1/len(graph[i])



    def page_rank(self, damping=0.85, limit=1.e-8):

        print(self.m)




