"""
Implement for line_graph(https://en.wikipedia.org/wiki/Line_graph)
This implementations are faster than nx.line_graph and memory efficient (when you have massive inputs).
"""

import pandas as pd
import numpy as np
import itertools
import argparse
import logging
from multiprocessing import Pool
import os


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="graph/karate-mirrored.edgelist")
    args = parser.parse_args()
    return args


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess for the transformation of edges into nodes"""

    data.columns = ["left", "right"]
    data.left = data.left.apply(str)
    data.right = data.right.apply(str)
    data["edge_name"] = data["left"] + "-" + data["right"]
    data["edge_name_int"] = np.arange(data.shape[0])

    return data


def paralize_convert_edge2node(args):
    data, unique_node_lists, node_name = args
    containing_edge_lists = data[(data.edge_name.str.startswith(node_name + "-"))
                                 | data.edge_name.str.endswith("-" + node_name)].edge_name_int
    with open("graph/{}-edges.edgelist".format(basename), "a") as f:
        for combination in itertools.combinations(containing_edge_lists, 2):
            f.write(str(combination[0])+" " + str(combination[1]) + "\n")


def convert_edge2node(data, unique_node_lists):
    args = [(data, unique_node_lists, node_name) for node_name in unique_node_lists]
    with Pool() as pool:
        pool.map(paralize_convert_edge2node, args)


def edgedata2nodedata(data: pd.DataFrame) -> pd.DataFrame:
    """Convert edge into node"""
    unique_node_lists = pd.concat([data.left, data.right], axis=0).unique()
    convert_edge2node(data, unique_node_lists)


def main(args, _logger):
    _logger.info("Read and Preprocess data")
    data = pd.read_csv(args.input, sep=" ", header=None)

    data = preprocess(data)
    global basename
    basename = args.input.split("/")[1].split(".")[0]
    if not os.path.exists("relations"):
        os.mkdir("relations")
    data.to_csv("relations/{}-edges_names.csv".format(basename), index=False, sep=" ")
    _logger.info("Start line-graph")
    if os.path.exists("graph/{}-edges.edgelist".format(basename)):
        os.remove("graph/{}-edges.edgelist".format(basename))
    edgedata2nodedata(data)
    _logger.info("Successfly saved")


if __name__ == '__main__':
    args = argument()
    _logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
    main(args, _logger)
