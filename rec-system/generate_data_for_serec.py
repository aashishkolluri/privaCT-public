import numpy as np
import os
from collections import defaultdict
from env import *
from preprocess import *

SERecDATADIR = "./serec_data/"


class GenSERecData(Preprocess):
    def __init__(
        self,
        dataset,
        ratings_file,
        trusts_file,
        seed_for_train_split,
        evaluate_cold_start_users=False,
        test_train_split_ratio=0.1,
        max_r=None,
        normalize=False,
    ):
        super(GenSERecData, self).__init__(
            ratings_file,
            trusts_file,
            seed_for_train_split,
            evaluate_cold_start_users,
            test_train_split_ratio,
            max_r,
            normalize,
        )
        self.datadir = SERecDATADIR + dataset + "/"
        self.gen_trust_file()
        self.gen_train_valid_test_data()

    def gen_trust_file(self):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        with open(self.datadir + "social_data", "w") as fp:
            for i in range(self.n_users):
                friends = np.argwhere(self.A[i] > 0.0)
                fp.write(str(len(friends)))
                if len(friends) == 0:
                    fp.write("\n")
                    continue
                for friend in friends:
                    fp.write(" " + str(friend) + ":1")
                fp.write("\n")

    def gen_train_valid_test_data(self):
        # get all tuples from self.train_user_item_rating_dict
        all_train_ratings = []
        for user in self.train_user_item_rating_dict:
            for each in self.train_user_item_rating_dict[user]:
                all_train_ratings.append((user, each[0], each[1]))

        # split them into validation data
        all_ratings_valid = self.prng.choice(
            range(len(all_train_ratings)), int(len(self.all_rating_tuples) * 0.2)
        )
        all_ratings_valid = [all_train_ratings[i] for i in all_ratings_valid]
        all_train_ratings_no_valid = list(
            set(all_train_ratings) - set(all_ratings_valid)
        )

        all_train_ratings_no_valid_dict = defaultdict(list)
        all_train_ratings_no_valid_dict_item_wise = defaultdict(list)
        all_ratings_valid_dict = defaultdict(list)

        for each in all_train_ratings_no_valid:
            # print(each)
            user = each[0]
            item = each[0]
            rating = each[2]
            all_train_ratings_no_valid_dict[user].append((item, rating))
            all_train_ratings_no_valid_dict_item_wise[item].append((user, rating))
        for each in all_ratings_valid:
            user = each[0]
            item = each[1]
            rating = each[2]
            all_ratings_valid_dict[user].append((item, rating))

        # generate output files
        with open(self.datadir + "train_user", "w") as fp:
            for i in range(self.n_users):
                user = i
                dictname = all_train_ratings_no_valid_dict
                if not user in dictname:
                    fp.write("0\n")
                    continue
                fp.write(str(len(dictname[user])))
                for each in dictname[user]:
                    fp.write(" " + str(each[0]) + ":" + str(int(each[1])))
                fp.write("\n")
        with open(self.datadir + "train_item", "w") as fp:
            for i in range(len(self.items)):
                item = i
                dictname = all_train_ratings_no_valid_dict_item_wise
                if not item in dictname:
                    fp.write("0\n")
                    continue
                fp.write(str(len(dictname[user])))
                for each in dictname[item]:
                    fp.write(" " + str(each[0]) + ":" + str(int(each[1])))
                fp.write("\n")
        with open(self.datadir + "test_data", "w") as fp:
            for i in range(self.n_users):
                user = i
                dictname = self.test_user_item_rating_dict
                if not user in dictname:
                    fp.write("0\n")
                    continue
                fp.write(str(len(dictname[user])))
                for each in dictname[user]:
                    fp.write(" " + str(each[0]) + ":" + str(int(each[1])))
                fp.write("\n")
        with open(self.datadir + "vali_data", "w") as fp:
            for i in range(self.n_users):
                user = i
                dictname = all_ratings_valid_dict
                if not user in dictname:
                    fp.write("0\n")
                    continue
                fp.write(str(len(dictname[user])))
                for each in dictname[user]:
                    fp.write(" " + str(each[0]) + ":" + str(int(each[1])))
                fp.write("\n")


if __name__ == "__main__":
    ratings_file = DATADIR + args.dataset + "/" + "ratings.txt"
    trusts_file = DATADIR + args.dataset + "/" + "trusts.txt"
    gen_serec = GenSERecData(args.dataset, ratings_file, trusts_file, args.seedSplit)
    print("Generated!")
