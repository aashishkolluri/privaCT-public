import multiprocessing
import utils_sim as US
import os
from collections import defaultdict
import dill as pickle

pickle.HIGHEST_PROTOCOL = 2
import numpy as np
import dp as DP


def findDegreeVec(n_partitions, eps_h, graph):
    nodes = graph.nodes()
    edges = graph.edges()
    N = len(nodes)
    # Get the trust users list and construct friends list
    users_trust = nodes
    friends = defaultdict(list)
    for each in edges:
        if not each[1] in friends[each[0]]:
            friends[each[0]].append(each[1])
        if not each[0] in friends[each[1]]:
            friends[each[1]].append(each[0])
    # partition the users and construct degree vectors
    np.random.shuffle(list(nodes))
    shuffled_users = nodes
    partitions = np.array_split(shuffled_users, n_partitions)
    partition_dict_k = defaultdict(int)
    for i in range(len(partitions)):
        for j in range(len(partitions[i])):
            partition_dict_k[partitions[i][j]] = i

    def _retVec():
        return np.zeros(n_partitions, dtype=np.int32)

    # dict of arrays {user: degree vector}
    degree_vec = defaultdict(_retVec)
    # construct the degree vectors each node
    for node in nodes:
        friends_list = friends[node]
        for friend in friends_list:
            grp_no = partition_dict_k[friend]
            degree_vec[node][grp_no] += 1

    if eps_h > 0.0:  # if need to add noise
        # Noise the degree vectors
        for each in degree_vec:
            degree_vec[each] = DP.histogramLDP(degree_vec[each], eps_h)
    # change the degree vector from dictionary to numpy array
    degree_vec_arr = np.zeros(shape=(N, n_partitions), dtype=np.float32)
    for key, value in degree_vec.items():
        degree_vec_arr[key] += value
    degree_vec = degree_vec_arr
    return degree_vec


# load or find similarities in degree vectors
def find_dissim(sim_file, degree_vec, users_trust, simalg="euclidean"):
    # similarities is a matrix of shape NxN where N is number of users in friends graph
    N = len(users_trust)
    similarities = np.zeros(shape=(N, N))
    if os.path.isfile(sim_file):
        with open(sim_file, "rb") as fp:
            _ = pickle.load(fp)
            similarities = pickle.load(fp)
        return similarities

    pool = multiprocessing.Pool(5)
    h = []
    for i in range(len(users_trust)):
        if simalg == "cosine":
            h.append(pool.apply_async(US.pcosine, args=(i, degree_vec)))
        elif simalg == "euclidean":
            h.append(pool.apply_async(US.euclidean, args=(i, degree_vec)))
        else:
            print("Did not implement the algorithm %s" % (simalg))
            exit()

    pool.close()
    pool.join()
    for e in h:
        for tup in e.get():
            similarities[tup[0]][tup[1]] = tup[2]

    print("Writing to: ", sim_file)
    with open(sim_file, "wb") as fp:
        pickle.dump(degree_vec, fp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(similarities, fp, pickle.HIGHEST_PROTOCOL)

    return similarities
