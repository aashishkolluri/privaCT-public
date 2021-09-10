import numpy as np
import networkx as nx
from anytree import Node, RenderTree
import math
import dill as pickle
import os
import multiprocessing

from env import *
from treefunc import *
import hrg as HRG
import findDistances as FD
import hrg_global as HRG_GLOBAL
import dp as DP


# print some graph statistics
def print_graph_stats(title, g):
    print("Simple stats for: " + title)
    print("Nodes: {:10f}".format(len(g.nodes())))
    print("Edges: {:10f}".format(len(g.edges())))
    print(
        "Density: {:10.4f}".format(
            (len(g.edges()) * 2.0) / (len(g.nodes()) * (len(g.nodes()) - 1))
        )
    )


def findSimilarities(DATADIR, cg, eps_l, seed, epochs, n_partitions):
    np.random.seed(seed)
    # get the degree vectors
    degree_vec = FD.findDegreeVec(n_partitions, eps_l, cg)
    # Saved model filename
    fileTreeDir = DATADIR + str(seed) + "/"
    if not os.path.exists(fileTreeDir):
        os.makedirs(fileTreeDir)
    # Find all the distances
    sim_file = (
        fileTreeDir
        + "dp"
        + "_eps_l_"
        + str(eps_l)
        + "_seed_"
        + str(seed)
        + "_epochs_"
        + str(epochs)
        + ".sim"
    )
    FD.find_dissim(sim_file, degree_vec, cg.nodes(), simalg="euclidean")
    return


# Learns a HCT
def createHCTmodel(
    DATADIR, cg, eps_l, seed, epochs, n_partitions, dp=True, globalHRG=False
):
    """
    DATADIR: data directory of the dataset, say for facebook <DATA_HOME>/facebook/
    cg: nx graph
    seed: seed to fix
    n_partitions: for degee vector

    Returns: a set of losses
    """

    np.random.seed(seed)
    # List of edges
    edges = list(cg.edges())
    # Removing the duplicates
    edgesSet = set(edges)
    # List of nodes
    array = list(cg.nodes())
    # Randomly shuffling the nodes
    np.random.shuffle(array)
    # duplicate edges as in undirected array
    duplicateEdgesSet = set()
    for each in edgesSet:
        tmp = (each[1], each[0])
        if not tmp in edgesSet:
            duplicateEdgesSet.add(tmp)
        duplicateEdgesSet.add(each)

    # Saved model filename
    fileTreeDir = DATADIR + str(seed) + "/"
    if not os.path.exists(fileTreeDir):
        os.makedirs(fileTreeDir)
    if not globalHRG:
        fileTreePath = (
            fileTreeDir
            + "dp"
            + "_eps_l_"
            + str(eps_l)
            + "_epochs_"
            + str(epochs)
            + ".pkl"
        )
    else:
        fileTreePath = (
            fileTreeDir
            + "dp_global"
            + "_eps_l_"
            + str(eps_l)
            + "_epochs_"
            + str(epochs)
            + ".pkl"
        )

    # Create a Tree Object or retrieve it from fle
    hrgTree = Tree(N, fileTreePath, seed)
    # Only do this if you have already not trained and saved the model
    if not os.path.isfile(fileTreePath):
        print("Tree not present. Creating a random tree")
        # Create random graph
        hrgTree.randomTree(
            hrgTree.treeRoot, array, N + 1, hrgTree.nodeNames, hrgTree.nodeMap
        )
    else:
        print("Tree present. Loading the tree")

    # create a HRGBuilder object by passing the Tree object
    if not globalHRG:
        hrgBuilder = HRG.HRGbuilder(hrgTree, dp, eps_l)
    else:
        hrgBuilder = HRG_GLOBAL.HRGbuilder(hrgTree, dp, eps_l)
    # get the degree vectors unless storeMem is True
    degree_vec = {}
    # Find all the distances
    similarities = {}
    sim_file = (
        fileTreeDir
        + "dp"
        + "_eps_l_"
        + str(eps_l)
        + "_seed_"
        + str(seed)
        + "_epochs_"
        + str(epochs)
        + ".sim"
    )
    # If decide to store the similarities in memory
    if not globalHRG:
        if os.path.isfile(sim_file):
            with open(sim_file, "rb") as f:
                degree_vec = pickle.load(f)
                similarities = pickle.load(f)
        else:
            degree_vec = FD.findDegreeVec(n_partitions, eps_l, cg)
            similarities = FD.find_dissim(
                sim_file, degree_vec, cg.nodes(), simalg="euclidean"
            )

    # Train the model
    print("Learning the tree...")
    thres = 0.01
    ldList, dsList, dasguptaList = hrgBuilder.learnDend(
        thres, N, duplicateEdgesSet, epochs, similarities, degree_vec, sim_file
    )
    # save the model
    hrgTree.saveModel(fileTreePath)

    return (ldList, dsList, dasguptaList)


if __name__ == "__main__":
    # Find the dataset directory
    DATADIR = DATA_HOME + args.dataset + "/"
    # Find the data file path containing graph edges
    datafilePath = DATADIR + args.dfPath
    DATASEEDDIR = DATA_HOME + args.dataset + "/" + str(args.seed) + "/"
    # Read the graph from the file depending on the format
    graph = {}
    if args.dataFormat == "txt":
        graph = nx.read_edgelist(datafilePath)
    elif args.dataFormat == "gml":
        graph = nx.read_gml(datafilePath)

    # Select the largest connected component if it is not connected
    cg = graph
    if not nx.is_connected(graph):
        cg = max(nx.connected_components(graph), key=len)
        cg = graph.subgraph(cg).copy()

    nodeMap = {}
    revNodeMap = {}
    # create new labels
    count = 0
    for node in cg.nodes():
        nodeMap[node] = count
        revNodeMap[count] = node
        count += 1
    # assign the new labels
    cg = nx.relabel_nodes(cg, nodeMap)

    N = len(cg.nodes())
    E = len(cg.edges())
    n_partitions = int(math.log(len(cg.nodes())))
    # Print the statistics of the graph
    print_graph_stats(args.dataset, cg)

    if args.createHCT:
        # The epsilon list
        epsList = [0.0, 0.25, 0.5, 1.0]
        # Check if need to build HRG graphs for all epsilons
        if args.all:
            logLDs = {}
            pool = multiprocessing.Pool(len(epsList))
            if args.onlydp:
                pool = multiprocessing.Pool(len(epsList) - 1)
            # Find the dissimilarity matrices for all parameters
            if not args.globalHRG:
                for each in epsList:
                    dp = True
                    if each == 0.0:
                        dp = False
                    findSimilarities(
                        DATADIR, cg, each, args.seed, args.epochs, n_partitions
                    )
            h = []
            for each in epsList:
                dp = True
                if each == 0.0:
                    dp = False
                h.append(
                    [
                        each,
                        pool.apply_async(
                            createHCTmodel,
                            args=(
                                DATADIR,
                                cg,
                                each,
                                args.seed,
                                args.epochs,
                                n_partitions,
                                dp,
                                args.globalHRG,
                            ),
                        ),
                    ]
                )
            pool.close()
            pool.join()
            for e in h:
                tup = e
                print(tup[0])
                logLDs[tup[0]] = tup[1].get()
            lossFile = DATASEEDDIR + "loss_all" + "_epochs_" + str(args.epochs) + ".pkl"
            if args.globalHRG:
                lossFile = DATASEEDDIR + "loss_global_all" + "_epochs_" + str(args.epochs) + ".pkl"
            with open(lossFile, "wb") as fp:
                pickle.dump(logLDs, fp, pickle.HIGHEST_PROTOCOL)
        else:
            dp = True
            if int(args.eps_l) == 0:
                dp = False
            ld, ds, das = createHCTmodel(
                DATADIR,
                cg,
                args.eps_l,
                args.seed,
                args.epochs,
                n_partitions,
                dp,
                args.globalHRG,
            )
            lossFile = (
                DATASEEDDIR
                + "loss_"
                + str(args.eps_l)
                + "_epochs_"
                + str(args.epochs)
                + ".pkl"
            )
            if args.globalHRG:
                lossFile = (
                    DATASEEDDIR
                    + "loss_global_"
                    + str(args.eps_l)
                    + "_epochs_"
                    + str(args.epochs)
                    + ".pkl"
                )
            with open(lossFile, "wb") as fp:
                pickle.dump({args.eps_l: (ld, ds, das)}, fp, pickle.HIGHEST_PROTOCOL)
