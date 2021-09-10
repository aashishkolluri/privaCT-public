import numpy as np
from scipy.stats import pearsonr as PR


def itemAvg(
    user,
    item,
    training_set,
    ratings_avg,
    user_item_rating_dict,
    users_for_same_item_dict,
    *args,
    **kwargs
):
    """
    Inputs:
    training_set: {user: set(items rated)}
    ratings_avg: {[user1_rating, user2_rating, ...,]]}
    M: user item rating matrix of training set

    Outputs:
    predicted rating for (user, item)
    """

    w_sum_rating = 0.0
    denom = 0.0
    # Get all the users with same item
    users_for_same_item = np.array(users_for_same_item_dict[item])
    # only consider the users_for_same_items in the training set for that item
    users_for_same_item_training = []

    for user1 in users_for_same_item:
        if (user1 in training_set) and (item in training_set[user1]):
            users_for_same_item_training.append(user1)

    users_for_same_item = np.array(users_for_same_item_training)
    if len(users_for_same_item) == 0:
        return (False, -99999999.0)
    # Get ratings of the similar users of that item
    rating = []
    for user1 in users_for_same_item:
        rating.append(user_item_rating_dict[(user1, item)])
    rating = np.array(rating)
    # Average ratings of all similar users
    user_avg_ratings = ratings_avg[users_for_same_item]
    w_sum_rating = np.sum(rating - user_avg_ratings)
    denom = len(rating)
    # Add the average of user rating on his items
    pred = ratings_avg[user] + w_sum_rating / (denom + 0.000000001)
    return (True, pred)


def friendsCF(
    user,
    item,
    training_set,
    ratings_avg,
    user_item_rating_dict,
    users_for_same_item_dict,
    *args,
    **kwargs
):
    """
    Inputs:
    training_set: (train_users_set, train_items_set)
    ratings_avg: {[user1_rating, user2_rating, ...,]]}
    user_item_rating_dict: {(user, item): rating}
    users_for_same_item_dict: {item:[user1, user2, ...,]}
    kwargs["A"]: adjacency matrix of the social network

    Outputs:
    predicted rating for (user, item)
    """

    w_sum_rating = 0.0
    denom = 0.0
    # Get all the users with same item
    users_for_same_item = np.array(users_for_same_item_dict[item])
    # only consider the users_for_same_items in the training set for that item
    users_for_same_item_training = []
    for user1 in users_for_same_item:
        if (user1 in training_set) and (item in training_set[user1]):
            users_for_same_item_training.append(user1)
    users_for_same_item = np.array(users_for_same_item_training)
    if len(users_for_same_item) == 0:
        return (False, -99999999.0)
    # Get ratings of the similar users of that item
    rating = []
    for user1 in users_for_same_item:
        rating.append(user_item_rating_dict[(user1, item)])
    rating = np.array(rating)
    A = kwargs["A"]
    sims = A[user][users_for_same_item]
    if np.sum(sims) == 0.0:
        return (
            False,
            itemAvg(
                user,
                item,
                training_set,
                ratings_avg,
                user_item_rating_dict,
                users_for_same_item_dict,
                *args,
                **kwargs
            )[1],
        )
    # Average ratings of all similar users
    user_avg_ratings = ratings_avg[users_for_same_item]
    w_sum_rating = np.sum((rating - user_avg_ratings) * sims)
    denom = np.sum(sims)
    # Add the average of user rating on his items
    pred = ratings_avg[user] + w_sum_rating / (denom + 0.000000001)
    return (True, pred)


def privaCTCF(
    user,
    item,
    training_set,
    ratings_avg,
    user_item_rating_dict,
    users_for_same_item_dict,
    *args,
    **kwargs
):
    """
    Inputs:
    training_set: (train_users_set, train_items_set)
    ratings_avg: {[user1_rating, user2_rating, ...,]]}
    user_item_rating_dict: {(user, item): rating}
    users_for_same_item_dict: {item:[user1, user2, ...,]}
    kwargs["similarities_HCT"]: {user:[user1, user2, user3, ...]}

    Outputs:
    predicted rating for (user, item)
    """
    # if not item in training_set[1]:
    #     return 0.0
    w_sum_rating = 0.0
    denom = 0.0
    # Get all the users with same item
    users_for_same_item = np.array(users_for_same_item_dict[item])
    # only consider the users_for_same_items in the training set for that item
    users_for_same_item_training = []
    for user1 in users_for_same_item:
        if (user1 in training_set) and (item in training_set[user1]):
            users_for_same_item_training.append(user1)
    users_for_same_item = np.array(users_for_same_item_training)

    if len(users_for_same_item) == 0:
        return (False, -99999999.0)

    # Get ratings of the similar users of that item
    rating = []
    for user1 in users_for_same_item:
        rating.append(user_item_rating_dict[(user1, item)])
    rating = np.array(rating)

    sims = []
    top_m_dict = kwargs["similarities_HCT"]
    for sim_user in users_for_same_item:
        sim = 0.0
        if sim_user in top_m_dict[user]:
            sim = 1.0
        sims.append(sim)

    if np.sum(sims) == 0.0:
        return (
            False,
            itemAvg(
                user,
                item,
                training_set,
                ratings_avg,
                user_item_rating_dict,
                users_for_same_item_dict,
                *args,
                **kwargs
            )[1],
        )

    # Average ratings of all similar users
    user_avg_ratings = ratings_avg[users_for_same_item]
    w_sum_rating = np.sum((rating - user_avg_ratings) * sims)

    denom = np.sum(sims)
    # Add the average of user rating on his items
    pred = ratings_avg[user] + w_sum_rating / (denom + 0.000000001)

    return (True, pred)
