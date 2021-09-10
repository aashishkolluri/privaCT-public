import numpy as np
import networkx as nx
from select_comms_hct import findallComs
from env import *
import operator
import sys

sys.path.append("../dpAlg/")
from treefunc import *
import dill as pickle

dataset = args.dataset
eps = args.eps_l
seed = args.seed

from preprocess import *
import algorithms
import metrics
from env import *
from scipy.stats import pearsonr as PR


def get_hct_degree_vec_filename(dataset, seed, eps=0.5):
    """
    this method will figure out what the filename should be based on a args.seed and args.eps
    For now, we hardcode the seed and the eps
    """
    # dp_eps_l_0.0_seed_1234_epochs_1000.sim
    return os.path.join(
        DATADIR,
        dataset,
        str(seed),
        "dp_eps_l_" + str(eps) + "_seed_" + str(seed) + "_epochs_1000.sim",
    )


degree_vec_file = get_hct_degree_vec_filename(args.dataset, args.seed)
with open(degree_vec_file, "rb") as fp:
    degree_vec = pickle.load(fp)


def privaCTCF(
    dataset,
    ratings_file,
    trusts_file,
    seed_for_train_split,
    K=100,
    evaluate_cold_start_users=True,
    test_train_split_ratio=0.1,
    seed=1234,
    eps=0.5,
    max_r=None,
):
    prep = Preprocess(
        ratings_file,
        trusts_file,
        seed_for_train_split,
        evaluate_cold_start_users,
        test_train_split_ratio,
        max_r,
    )
    ndcgs, maps = [], []
    results = {}
    result_file_name = (
        dataset
        + "_privaCTCF_"
        + "_seed_"
        + str(seed_for_train_split)
        + "_hct_seed_"
        + str(seed)
        + ".out"
    )
    result_file_name.split(".out")[0] + "_cold" + ".out"
    with open(RESULT_DIR + result_file_name, "rb") as fp:
        results = pickle.load(fp)

    ndcg_dict = {}
    ndcg = metrics.NDCG(prep.test_user_item_rating_dict, results, K, ndcg_dict)
    #        MAP = metrics.MAP(prep.test_user_item_rating_dict, results, K)
    #        ndcgs.append(ndcg)
    #        maps.append(MAP)
    print("NDCG: ", ndcg)

    return (prep, results, ndcg_dict)


ratings_file = DATADIR + args.dataset + "/" + "ratings.txt"
trusts_file = DATADIR + args.dataset + "/" + "trusts.txt"
seed_for_train_splits = [67, 13, 45, 7, 82]
for seed_for_train_split in seed_for_train_splits:
    # querying for the ndcg scores
    prep, results, ndcg_dict = privaCTCF(
        args.dataset, ratings_file, trusts_file, seed_for_train_split, seed=args.seed
    )

    m = 1
    # if len(sys.argv)>4:
    #   m = sys.argv[4]
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    def getDistances(node, n2nDict, sp, k=-1):
        ds = []
        for temp in list(n2nDict[int(node)][:k]):
            ds.append(sp[node][str(temp)])
        return ds

    # get the graph
    gname = DATADIR + dataset + "/" + "trusts.txt"
    g = nx.read_edgelist(gname)
    degs_pairs = list(nx.degree(g))
    # degs = [x[1] for x in degs_pairs]
    # deg_max = max(degs)
    # print(deg_max)

    # First compute the shortest path profiles
    tfname = (
        DATADIR
        + dataset
        + "/"
        + str(seed)
        + "/"
        + "dp_eps_l_"
        + str(eps)
        + "_epochs_1000.pkl"
    )
    hrgTree = Tree(0, tfname, int(seed))
    n2nDict = defaultdict(list)
    findallComs(hrgTree.treeRoot, n2nDict)

    # all shortest paths
    sp = dict(nx.all_pairs_shortest_path_length(g))
    mms = []

    # node: [n1, n2, n3] ----> node: [d1, d2, d3...,]
    for node in g.nodes():
        mms.append(getDistances(node, n2nDict, sp, len(n2nDict)))

    sorted_ndcg_list = sorted(ndcg_dict.items(), key=operator.itemgetter(1))
    xlist, ylist = [], []
    count = 0
    sps = []
    ndcgs = []
    # for node, ndcg in sorted_ndcg_list:
    #     print(node, ndcg, nx.degree(g, str(node)), np.mean(mms[node][:nx.degree(g, str(node))]))
    #     ndcgs.append(ndcg)
    #     sps.append(np.mean(mms[node][:nx.degree(g, str(node))]))
    # print("PR: ", PR(ndcgs, sps))
    # print(sorted_ndcg_list)
    for node, ndcg in sorted_ndcg_list:
        # print(sorted_ndcg_list)
        # if ndcg > 0.0:
        xlist.append(count)
        ndcgs.append(ndcg)
        count += 1
        tmp = []

        d = degree_vec[node]
        d1 = int(np.sum(d))
        if d1 <= 0:
            d1 = 0
        # if node==148:
        # print(results[node][:100], prep.test_user_item_rating_dict[node])
        for item, rating in results[node][:100]:
            tmp1 = []
            for nei in list(n2nDict[int(node)][:d1]):
                if nei in prep.train_user_item_dict:
                    if item in set(prep.train_user_item_dict[nei]):
                        tmp1.append(sp[str(node)][str(nei)])
            if len(tmp1) > 0:
                tmp.append(np.mean(tmp1))
            else:
                continue
        if len(tmp) > 0:
            ylist.append(np.mean(tmp))
        else:
            ndcgs = ndcgs[:-1]
            xlist = xlist[:-1]
            count -= 1
    # print(ndcgs, ylist)
    print("seed:", str(args.seed), " PR: ", PR(ndcgs, ylist))
    np.savetxt(
        "sp_dist_final_ndcg_" + dataset + ".csv",
        np.stack([xlist, ylist]).T,
        delimiter=",",
        header="ind,dist",
        comments="",
    )
