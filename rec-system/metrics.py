import numpy as np
import operator
import math


def NDCG(gt_dict, pred_dict, K=100, ndcg_dict=None):
    """
    Inputs:
    gt_dict: {user: [(item, score)] sorted in decreasing order}
    pred_dict: {user: [(item, score)] sorted in decreasing order}

    Outputs:
    Average NDCG across all users
    """
    if not (len(pred_dict) == len(gt_dict)):
        print(" The ground truth and predicted are of different lengths! ")
        exit()

    # construct gt item sets for each user
    gt_user_items = {}
    for user in gt_dict:
        gt_user_items[user] = set()
        for x in gt_dict[user]:
            if x[1] > 0.0:
                gt_user_items[user].add(x[0])

    sum_NDCG = 0.0
    for user in pred_dict:
        dcg = 0.0
        idcg = 0.0
        # Take only first K items,
        # Assuming that the items in pred_dict[i] are more than K
        for pos, item in enumerate(pred_dict[user][:K]):
            if item[0] in gt_user_items[user]:
                dcg += 1.0 / math.log(pos + 2, 2)

        for pos, item in enumerate(gt_dict[user][:K]):
            idcg += 1.0 / math.log(pos + 2, 2)

        sum_NDCG += dcg / idcg
        if not ndcg_dict == None:
            ndcg_dict[user] = dcg / idcg

    return sum_NDCG / len(pred_dict)


def MAP(gt_dict, pred_dict, K=100):
    """
    Inputs:
    gt_dict: {user: [(item, score)] sorted in decreasing order}
    pred_dict: {user: [(item, score)] sorted in decreasing order}

    Outputs:
    MAP across all users
    """
    if not (len(pred_dict) == len(gt_dict)):
        print(" The ground truth and predicted are of different lengths! ")
        exit()

    # construct gt item sets for each user
    gt_user_items = {}
    for user in gt_dict:
        gt_user_items[user] = set()
        for x in gt_dict[user]:
            if x[1] > 0.0:
                gt_user_items[user].add(x[0])

    sum_prec = 0.0
    for user in pred_dict:
        hits = 0.0
        prec = 0.0
        # Take only first K items,
        # Assuming that the items in pred_dict[i] are more than K
        for pos, item in enumerate(pred_dict[user][:K]):
            if item[0] in gt_user_items[user]:
                hits += 1.0
                prec += hits / (pos + 1.0)
        sum_prec += prec / (min(len(gt_user_items[user]), K) + 0.000000001)

    return sum_prec / len(pred_dict)


if __name__ == "__main__":
    gt_dict = {
        1: [("a", 1.0), ("b", 1.0), ("c", 9.0), ("d", 0.0)],
        2: [("a", 9.0), ("d", 0.5)],
    }
    pred_dict = {
        1: [("e", 8.3), ("c", 2.0), ("a", 1.0), ("f", 0.0)],
        2: [("b", 0.9), ("a", 2.0)],
    }
    print(f"NDCG top 5:{NDCG(gt_dict, pred_dict, 5)}")
    print(f"NDCG top 2:{NDCG(gt_dict, pred_dict, 2)}")
    print(f"MAP top 5:{MAP(gt_dict, pred_dict, 5)}")
    print(f"MAP top 2:{MAP(gt_dict, pred_dict, 2)}")
