import sys
import os

sys.path.append("../dpAlg")
from treefunc import *
from env import *


def findallComs(root, n2nDict):
    queue = deque()
    # list of top influnetial nodes
    # Do BFS
    queue.append(root)
    while not len(list(queue)) == 0:
        node = queue.popleft()
        if len(node.children) == 2:
            for ln in node.left:
                if n2nDict[ln] == None:
                    n2nDict[ln] = []
                n2nDict[ln].extend(node.right)
            for rn in node.right:
                if n2nDict[rn] == None:
                    n2nDict[rn] = []
                n2nDict[rn].extend(node.left)
            queue.append(node.children[0])
            queue.append(node.children[1])

    for each in n2nDict:
        n2nDict[each].reverse()


def select_closest_hct(node, degree_vec, n2nDict):
    inodes = []
    d = degree_vec[node]
    degree = int(np.sum(d))
    if degree <= 0:
        degree = 0
    nodeList = n2nDict[node]
    inodes.extend(nodeList[: min(degree, len(degree_vec))])

    return inodes


if __name__ == "__main__":
    dataset = args.dataset
    eps = args.eps_l
    seed = args.seed
    degree_vec_file = os.path.join(
        DATADIR,
        dataset,
        str(seed),
        "dp_eps_l_" + str(eps) + "_seed_" + str(seed) + "_epochs_1000.sim",
    )

    hct_file = os.path.join(
        DATADIR, dataset, str(seed), "dp_eps_l_" + str(eps) + "_epochs_1000.pkl"
    )

    sim_hct = {}
    n2nDict = defaultdict(list)  # {user: [f1, f2, f3,...] fi is in community}
    with open(degree_vec_file, "rb") as fp:
        degree_vec = pickle.load(fp)
    hrgTree = Tree(0, hct_file, seed)
    if len(n2nDict) == 0:
        print("Constructing the communities")
        findallComs(hrgTree.treeRoot, n2nDict)
    if len(n2nDict) == 0:
        print("Cannot construct communities from HCT")
        exit()

    for i in range(len(degree_vec)):
        user = i
        sim_hct[user] = set(select_closest_hct(user, degree_vec, n2nDict))

    with open(
        RESULT_DIR
        + "trusts_dp_dataset_"
        + dataset
        + "eps_l_"
        + str(eps)
        + "_seed_"
        + str(seed)
        + "_epochs_1000_knn.txt",
        "w",
    ) as fp:
        for user in sim_hct:
            for user1 in list(sim_hct[user]):
                fp.write(str(user) + "\t" + str(user1) + "\n")
                fp.write(str(user1) + "\t" + str(user) + "\n")
