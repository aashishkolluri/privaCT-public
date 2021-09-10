import numpy as np
import sys

filename = sys.argv[1]
r_f, r_i, r_p = [], [], []
with open(filename, "r") as fp:
    lines = fp.readlines()
    for line in lines:
        if "---" in line:
            halves = line.split(" --- ")
            words = halves[1].split(", ")
            ndcg = float(words[0].split(" : ")[1])
            MAP = float(words[1].split(" : ")[1])
            if "friends" in halves[0]:
                r_f.append((ndcg, MAP))
            if "itemAvg" in halves[0]:
                r_i.append((ndcg, MAP))
            if "priva" in halves[0]:
                r_p.append((ndcg, MAP))


print(
    "friendsCF NDCG : {}, Map : {}".format(
        np.mean([x[0] for x in r_f]), np.mean([x[1] for x in r_f])
    )
)
print(
    "itemAvg NDCG : {}, Map : {}".format(
        np.mean([x[0] for x in r_i]), np.mean([x[1] for x in r_i])
    )
)
print(
    "privaCTCF NDCG : {}, Map : {}".format(
        np.mean([x[0] for x in r_p]), np.mean([x[1] for x in r_p])
    )
)
