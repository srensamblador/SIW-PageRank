from common import Graph
import argparse

def load_graph_from_file(filename):
    with open(filename) as f:
        edges = []
        for line in f:
            edges.append(line.strip().split(","))
        return Graph(edges)

def main(args):
    graph = load_graph_from_file(args.file)
    scores = graph.page_rank(float(args.damping), float(args.limit))
    [print(node, score) for node, score in sorted(scores.items(), key=lambda i: i[1], reverse=True)]


def parse_args():
    parser = argparse.ArgumentParser(description="Loads graph from file")
    parser.add_argument("file", help="File containing the edges")
    parser.add_argument("-d", "--damping", help="Damping factor")
    parser.add_argument("-l", "--limit", default=1e-8, help="min quadratic mean error at which PageRank should stop iterating")
    args = parser.parse_args()
    return args


main(parse_args())