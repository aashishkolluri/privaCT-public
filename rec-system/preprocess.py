import numpy as np
import networkx as nx
import operator
from scipy.stats import pearsonr
import community as CL
import pickle
from collections import defaultdict

import sys

sys.path.append("../dpAlg")
from treefunc import *
from select_comms_hct import *


class Preprocess:
    def __init__(
        self,
        ratings_file,
        trusts_file,
        seed_for_train_split,
        evaluate_cold_start_users=True,
        test_train_split_ratio=0.1,
        max_r=None,
        normalize=True,
    ):
        """
        Inputs:
        ratings_file: the full path to the ratings file
        trusts_file: the full path to the trusts file
        seed_for_train_split: seed for the train test split
        """

        self.trusts_file = trusts_file
        self.seed_for_train_split = seed_for_train_split
        self.max_r = max_r
        self.prng = np.random.default_rng(self.seed_for_train_split)
        self.evaluate_cold_start_users = evaluate_cold_start_users
        self.A = self.loadTrusts(trusts_file)  # the adjacency matrix of trusts
        self.n_users = len(self.A)  # total number of users
        self.users = set(range(self.n_users))  # all users
        (
            self.M,  # a matrix with the ratings (self.n_users, len(self.items))
            self.user_item_rating_dict,
            self.items,  # a set of all items
            self.users_with_ratings,  # set of users with atleast one rating
            self.all_rating_tuples,  # [(user, item, rating)]
            self.users_for_same_item_dict,
            self.items_for_same_user_dict,
        ) = self.loadRatings(ratings_file, normalize)
        self.users_with_no_ratings = self.users - self.users_with_ratings
        (
            self.train_user_item_rating_dict,
            self.test_user_item_rating_dict,
        ) = self.trainTestSplit(test_train_split_ratio)
        self.train_user_item_dict, self.ratings_avg = self.computeAuxillaryDS()

    def normalizeRatings(self, arr):
        if not self.max_r == None:
            return arr / (self.max_r + 0.000000001)
        else:
            return arr / (arr.max() + 0.000000001)

    def loadTrusts(self, trusts_file):
        """
        Outputs:
        A: an adjacency matrix of trusts
        """
        # Get number of trust users
        n_users = len(nx.read_edgelist(trusts_file).nodes())
        A = np.zeros((n_users, n_users), dtype=np.float32)
        with open(trusts_file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                items = line.split()
                A[int(items[0])][int(items[1])] = 1.0
                A[int(items[1])][int(items[0])] = 1.0
        return A

    def loadRatings(self, ratings_file, normalize=True):
        """
        Loads the ratings
        """
        items = set()
        users_with_ratings = set()
        ratings = []
        ratings_dict = {}
        user_item_rating_dict = {}
        users_for_same_item_dict = {}
        items_for_same_user_dict = {}
        with open(ratings_file, "r") as fp:
            for line in fp.readlines():
                words = line.split()
                user = int(words[0])
                item = int(words[1])
                rating = float(words[2])
                items.add(item)
                users_with_ratings.add(user)
                if not user in ratings_dict:
                    ratings_dict[user] = []
                    ratings_dict[user].append([item])
                    ratings_dict[user].append([rating])
                else:
                    ratings_dict[user][0].append(item)
                    ratings_dict[user][1].append(rating)
                user_item_rating_dict[(user, item)] = rating
                if not item in users_for_same_item_dict:
                    users_for_same_item_dict[item] = []
                users_for_same_item_dict[item].append(user)
                if not user in items_for_same_user_dict:
                    items_for_same_user_dict[user] = []
                items_for_same_user_dict[user].append(item)

        if len(users_with_ratings) < self.n_users:
            print("There are additional users in trusts file compared to ratings file!")
            print("Users: ", list(self.users - users_with_ratings))
            # exit()

        n_items = len(items)
        M = np.full((self.n_users, n_items), 0.0)
        # normalization of ratings
        for user in ratings_dict:
            ratings_user = ratings_dict[user][1]
            arr = np.array(ratings_user)
            if normalize:
                arr = self.normalizeRatings(arr)
            for i in range(len(ratings_dict[user][0])):
                item = ratings_dict[user][0][i]
                if arr[i] == 0.0:
                    arr[i] += 0.000000001
                M[user][item] = arr[i]
                user_item_rating_dict[(user, item)] = arr[i]

        ratings = []
        # construct the ratings list
        for i in range(self.n_users):
            user = i
            items_user = ratings_dict[user][0]
            for item in items_user:
                ratings.append((user, item, user_item_rating_dict[(user, item)]))

        return (
            M,
            user_item_rating_dict,
            items,
            users_with_ratings,
            ratings,
            users_for_same_item_dict,
            items_for_same_user_dict,
        )

    def trainTestSplit(self, test_train_split_ratio):
        """
        Splits the ratings into training and test datasets
        """
        if self.evaluate_cold_start_users:
            test_users_for_cold_start = self.prng.choice(
                list(self.users_with_ratings),
                int(len(self.users_with_ratings) * test_train_split_ratio),
            )
            train_users_for_cold_start = self.users_with_ratings - set(
                test_users_for_cold_start
            )
            train_user_item_rating_dict = {}  # {user: [(item, rating)]} of train data
            test_user_item_rating_dict = {}  # {user: [(item, rating)]} of test data

            for user in train_users_for_cold_start:
                train_user_item_rating_dict[user] = []
                items_for_user = self.items_for_same_user_dict[user]
                for item in items_for_user:
                    train_user_item_rating_dict[user].append(
                        (item, self.user_item_rating_dict[(user, item)])
                    )
                train_user_item_rating_dict[user] = sorted(
                    train_user_item_rating_dict[user],
                    key=operator.itemgetter(1),
                    reverse=True,
                )

            for user in test_users_for_cold_start:
                test_user_item_rating_dict[user] = []
                items_for_user = self.items_for_same_user_dict[user]
                for item in items_for_user:
                    test_user_item_rating_dict[user].append(
                        (item, self.user_item_rating_dict[(user, item)])
                    )
                test_user_item_rating_dict[user] = sorted(
                    test_user_item_rating_dict[user],
                    key=operator.itemgetter(1),
                    reverse=True,
                )

            return (train_user_item_rating_dict, test_user_item_rating_dict)

        test_indices = self.prng.choice(
            range(len(self.all_rating_tuples)),
            int(len(self.all_rating_tuples) * test_train_split_ratio),
        )
        test_rating_tuples = [self.all_rating_tuples[i] for i in test_indices]
        # print(type(self.all_rating_tuples), test_rating_tuples)
        train_rating_tuples = set(self.all_rating_tuples) - set(test_rating_tuples)
        train_user_item_rating_dict = {}
        test_user_item_rating_dict = {}

        for each_tuple in train_rating_tuples:
            user = each_tuple[0]
            item = each_tuple[1]
            rating = each_tuple[2]

            if not user in train_user_item_rating_dict:
                train_user_item_rating_dict[user] = []
            train_user_item_rating_dict[user].append((item, rating))
        for user, ulist in train_user_item_rating_dict.items():
            train_user_item_rating_dict[user] = sorted(
                train_user_item_rating_dict[user],
                key=operator.itemgetter(1),
                reverse=True,
            )

        for each_tuple in test_rating_tuples:
            user = each_tuple[0]
            item = each_tuple[1]
            rating = each_tuple[2]

            if not user in test_user_item_rating_dict:
                test_user_item_rating_dict[user] = []
            test_user_item_rating_dict[user].append((item, rating))
        for user, ulist in test_user_item_rating_dict.items():
            test_user_item_rating_dict[user] = sorted(
                test_user_item_rating_dict[user],
                key=operator.itemgetter(1),
                reverse=True,
            )

        return (train_user_item_rating_dict, test_user_item_rating_dict)

    def computeAuxillaryDS(self):
        """
        Computes auxillary data structures based on training data

        Output:
        1) {user: {item,}} 2) an array of average ratings with user indexes
        """
        user_item_dict = {}
        avg_rating = np.full((self.n_users,), 0.0)
        for user in self.train_user_item_rating_dict:
            avg_rating[user] = np.mean(
                [x[1] for x in self.train_user_item_rating_dict[user]]
            )
            user_item_dict[user] = set(
                [x[0] for x in self.train_user_item_rating_dict[user]]
            )
        return (user_item_dict, avg_rating)

    def forCommunityCF(self):
        """
        Computes communities

        Output:
        {node: com}
        """
        cg = nx.read_edgelist(self.trusts_file)
        partition = CL.best_partition(cg)
        partition_int = {}
        for user, com in partition.items():
            partition_int[int(user)] = com
        return partition_int

    def forPrivaCTCF(self, sim_file, degree_vec_file, hct_file, seed):
        """
        Compute the closest users to each user based on HCT

        Output:
        {user: set(closest users)}
        """
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
        for user in self.users:
            sim_hct[user] = set(select_closest_hct(user, degree_vec, n2nDict))
        return sim_hct
