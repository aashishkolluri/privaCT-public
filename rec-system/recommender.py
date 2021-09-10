import operator
import pickle
import os
import time

from preprocess import *
import algorithms
import metrics
from env import *


def get_hct_sim_filename(dataset, seed, eps=0.5):
    """
    this method will figure out what the filename should be based on a args.seed and args.eps
    For now, we hardcode the seed and the eps
    """
    return os.path.join(
        DATADIR,
        dataset,
        str(seed),
        "dp_eps_l_" + str(eps) + "_epochs_1000_sim_file.pkl",
    )


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


def get_hct_pkl_filename(dataset, seed, eps=0.5):
    """
    this method will figure out what the filename should be based on a args.seed and args.eps
    For now, we hardcode the seed and the eps
    """
    return os.path.join(
        DATADIR, dataset, str(seed), "dp_eps_l_" + str(eps) + "_epochs_1000.pkl"
    )


def saveResults(result_file_name, results):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    with open(RESULT_DIR + result_file_name, "wb") as fp:
        pickle.dump(results, fp, protocol=2)


def predictForColdStart(method, prep, kwargs):
    start_time = time.time()
    results = {}
    print("Starting to predict for cold start")
    for user in prep.test_user_item_rating_dict:
        results[user] = []
        pos_results = []
        neg_results = []
        for item in prep.items:
            pred = method(
                user,
                item,
                prep.train_user_item_dict,
                prep.ratings_avg,
                prep.user_item_rating_dict,
                prep.users_for_same_item_dict,
                **kwargs,
            )
            if pred[0]:
                pos_results.append((item, pred[1]))
            else:
                neg_results.append((item, pred[1]))
        results[user] = sorted(
            pos_results, key=operator.itemgetter(1), reverse=True
        ) + sorted(neg_results, key=operator.itemgetter(1), reverse=True)
    print(f"Done predicting for cold start: {time.time()-start_time}")
    return results


# Obsolete
def predict(method, prep, kwargs):
    results = {}
    users_not_in_train_dict = []
    for user in prep.test_user_item_rating_dict:
        results[user] = []
        for item in prep.items:
            if not user in prep.train_user_item_dict:
                users_not_in_train_dict.append(user)
                continue
            if not item in prep.train_user_item_dict[user]:
                pred = method(
                    user,
                    item,
                    prep.train_user_item_dict,
                    prep.ratings_avg,
                    prep.M,
                    **kwargs,
                )
                results[user].append((item, pred))
        results[user] = sorted(results[user], key=operator.itemgetter(1), reverse=True)
    for user in set(users_not_in_train_dict):
        prep.test_user_item_rating_dict.pop(user)
    return results


def itemAvg(
    dataset,
    ratings_file,
    trusts_file,
    seed_for_train_split,
    K=100,
    evaluate_cold_start_users=True,
    test_train_split_ratio=0.1,
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
    results = {}
    kwargs = {}
    result_file_name = (
        dataset + "_itemAvg_" + "_seed_" + str(seed_for_train_split) + ".out"
    )
    if evaluate_cold_start_users:
        results = predictForColdStart(algorithms.itemAvg, prep, kwargs)
        result_file_name.split(".out")[0] + "_cold" + ".out"
    else:
        results = predict(algorithms.itemAvg, prep, kwargs)

    # ==== Uncomment if you want to save your results. Memory Intensive! ==== #
    # saveResults(result_file_name, results)

    return (
        metrics.NDCG(prep.test_user_item_rating_dict, results, K),
        metrics.MAP(prep.test_user_item_rating_dict, results, K),
    )


def friendsCF(
    dataset,
    ratings_file,
    trusts_file,
    seed_for_train_split,
    K=100,
    evaluate_cold_start_users=True,
    test_train_split_ratio=0.1,
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
    results = {}
    kwargs = {}
    kwargs["A"] = prep.A
    result_file_name = (
        dataset + "_friendsCF_" + "_seed_" + str(seed_for_train_split) + ".out"
    )
    if evaluate_cold_start_users:
        results = predictForColdStart(algorithms.friendsCF, prep, kwargs)
        result_file_name.split(".out")[0] + "_cold" + ".out"
    else:
        results = predict(algorithms.friendsCF, prep, kwargs)

    # ==== Uncomment if you want to save your results. Memory Intensive! ==== #
    # saveResults(result_file_name, results)

    return (
        metrics.NDCG(prep.test_user_item_rating_dict, results, K),
        metrics.MAP(prep.test_user_item_rating_dict, results, K),
    )


def privaCTCF(
    dataset,
    ratings_file,
    trusts_file,
    seed_for_train_split,
    K=100,
    evaluate_cold_start_users=True,
    test_train_split_ratio=0.1,
    seeds=[1234, 144, 1766],
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
    for seed in seeds:
        results = {}
        kwargs = {}
        kwargs["similarities_HCT"] = prep.forPrivaCTCF(
            get_hct_sim_filename(dataset, seed, eps),
            get_hct_degree_vec_filename(dataset, seed, eps),
            get_hct_pkl_filename(dataset, seed, eps),
            seed,
        )
        result_file_name = (
            dataset
            + "_privaCTCF_"
            + "_seed_"
            + str(seed_for_train_split)
            + "_hct_seed_"
            + str(seed)
            + ".out"
        )
        if evaluate_cold_start_users:
            results = predictForColdStart(algorithms.privaCTCF, prep, kwargs)
            result_file_name.split(".out")[0] + "_cold" + ".out"
        else:
            results = predict(algorithms.privaCTCF, prep, kwargs)

        # ==== Uncomment if you want to save your results. Memory Intensive! ==== #
        # saveResults(result_file_name, results)

        ndcg = metrics.NDCG(prep.test_user_item_rating_dict, results, K)
        mAP = metrics.MAP(prep.test_user_item_rating_dict, results, K)
        ndcgs.append(ndcg)
        maps.append(mAP)

    return (np.mean(ndcgs), np.mean(maps))


if __name__ == "__main__":
    methodDict = {"privaCTCF": privaCTCF, "friendsCF": friendsCF, "itemAvg": itemAvg}
    ratings_file = DATADIR + args.dataset + "/" + "ratings.txt"
    trusts_file = DATADIR + args.dataset + "/" + "trusts.txt"

    if args.method == "all":
        # Need to implement multiprocessing
        pass
    else:
        if not args.method in methodDict:
            print("Method not correct. See python recommender.py --h")
        else:
            methodcall = methodDict[args.method]
            ndcg, MAP = 0.0, 0.0
            if methodcall == privaCTCF:
                if args.seed > -1:
                    ndcg, MAP = methodcall(
                        args.dataset,
                        ratings_file,
                        trusts_file,
                        args.seedSplit,
                        args.K,
                        args.cold,
                        args.tSplitRatio,
                        [args.seed],
                        args.eps_l,
                        args.max_r,
                    )
                else:
                    ndcg, MAP = methodcall(
                        args.dataset,
                        ratings_file,
                        trusts_file,
                        args.seedSplit,
                        args.K,
                        args.cold,
                        args.tSplitRatio,
                        eps=args.eps_l,
                        max_r=args.max_r,
                    )
            else:
                ndcg, MAP = methodcall(
                    args.dataset,
                    ratings_file,
                    trusts_file,
                    args.seedSplit,
                    args.K,
                    args.cold,
                    args.tSplitRatio,
                    args.max_r,
                )
            print(
                f"Dataset {args.dataset}, Method {args.method}, top {args.K}, seed {args.seedSplit} --- NDCG : {ndcg}, MAP : {MAP}"
            )
