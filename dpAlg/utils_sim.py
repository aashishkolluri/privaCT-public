import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cosine similarity between degree vectors
def pcosine(i, degree_vec):
    mlist = []
    for j in range(i, len(degree_vec)):
        if i == j:
            sim = 1.0
            mlist.append((i, j, sim))
        else:
            degree_vec[i] = degree_vec[i].clip(min=0)
            degree_vec[j] = degree_vec[j].clip(min=0)
            sim = cosine_similarity(degree_vec[i][None, :], degree_vec[j][None, :])
            mlist.append((i, j, sim))
            mlist.append((j, i, sim))

    if i % 1000 == 0:
        print("Done with {:<20}".format(i))

    return mlist


# Euclidean distance based similarity between degree vectors
def euclidean(i, degree_vec):
    mlist = []
    for j in range(i + 1, len(degree_vec)):
        sim = max(1.0, np.linalg.norm(degree_vec[i] - degree_vec[j], ord=1))
        mlist.append((i, j, sim))
        mlist.append((j, i, sim))

    if i % 1000 == 0:
        print("Done with {:<20}".format(i))
    return mlist


def normalizeSim(similarities):
    # Normalize the similarities array
    for i in range(len(similarities)):
        x_max = np.max(similarities[i])
        x_min = np.min(similarities[i])
        similarities[i] = (similarities[i] - x_min) / (x_max - x_min)


# Assume edges format is set of tuples
def findIntersection(lista, listb, edges):
    cartProd = [(a, b) for a in lista for b in listb]
    intersection = edges.intersection(set(cartProd))
    return len(intersection)


def dissimilarityAvgDist(lista, listb, similarities):
    # Find all distances
    sumDistance = 0.0
    indices = np.transpose([np.tile(lista, len(listb)), np.repeat(listb, len(lista))])
    sumDistance = np.sum(similarities[indices[:, 0], indices[:, 1]])

    if sumDistance == 0.0:
        return 0.0000001
    return sumDistance / (len(lista) * len(listb))