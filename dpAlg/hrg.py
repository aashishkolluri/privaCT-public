import utils_sim as UTILS_SIM
import math
import time
import numpy as np
import dp as DP


class HRGbuilder:
    def __init__(self, treeObject, dp=False, eps_l=0.5):
        self._treeObj = treeObject
        self._dp = dp
        self._eps_l = eps_l

    # calculate the likelihood once for the entire tree
    def _likelihood(self, duplicateEdgesSet, similarities=[], degree_vec={}):
        """
            calculates the likehood and dissmilarity once
        """
        nodeNames = self._treeObj.nodeNames
        nodeMap = self._treeObj.nodeMap
        logld = 0.0
        logds = 0.0
        logdasgupta = 0.0
        for each in nodeNames:
            n_r = len(nodeMap[each].right)
            n_l = len(nodeMap[each].left)
            inter = UTILS_SIM.findIntersection(
                nodeMap[each].left, nodeMap[each].right, duplicateEdgesSet
            )
            p_r = (inter * 1.0) / (n_r * n_l)
            if p_r <= 0.0:
                p_r = 0.000000001
            if p_r >= 1.0:
                p_r = 0.999999999

            logld = (
                logld + inter * math.log(p_r) + (n_r * n_l - inter) * math.log(1 - p_r)
            )
            nodeMap[each]._pr = p_r
            nodeMap[each]._interEdges = inter

            """
                dissimilarity
            """
            dissim = UTILS_SIM.dissimilarityAvgDist(
                nodeMap[each].left, nodeMap[each].right, similarities
            )
            nodeMap[each]._dissim = dissim * (n_r * n_l) * (n_r + n_l)
            logds += dissim
            logdasgupta += dissim * n_r * n_l * (n_r + n_l)
        print("Started with likelihood: {}".format(logld))
        return (logld, logds, logdasgupta)

    # Define stop condition for MCMC
    def _stopCondition(self, ldList, epoch, N, k=5, thres=0.0001):
        if epoch < 2 * k:
            return False
        #     print(ldList)
        interval = k * N  # k epochs
        lastkList = ldList[epoch - 2 * k : epoch - k]
        currkList = ldList[epoch - k : epoch]
        lastkMean = (sum(lastkList) * 1.0) / k
        currkMean = (sum(currkList) * 1.0) / k
        if (max(currkMean - lastkMean, lastkMean - currkMean)) / lastkMean < thres:
            return True
        return False

    def _sanitize(self, value):
        if value <= 0.0:
            return 0.000000001
        if value >= 1.0:
            return 0.999999999
        return value

    # Metropolis Hastings based algorithm
    def learnDend(
        self,
        thres,
        N,
        duplicateEdgesSet,
        epochs,
        similarities=[],
        degree_vec={},
        sim_file=None,
    ):
        """
            In each iteration
                1) Select a node at random
                2) Select one of the two local configurations of the tree at random
                3) change the tree with probability min(1, LD_new/LD_old)
        """
        nodeNames = self._treeObj.nodeNames
        nodeMap = self._treeObj.nodeMap
        # First assign labels
        self._treeObj.assignLabel()
        # Make sure that the internal nodes satisfy ordering property i.e.,
        # node.label = node.left.label where node.label is the minimum leaf number in its subtree
        self._treeObj.makeOrPTree()

        # Calculate likelihood for the first time
        LD_old, logds_old, logdasgupta_old = self._likelihood(
            duplicateEdgesSet, similarities, degree_vec
        )
        # Start learning HRG
        # store lds
        ldList = []
        dsList = []
        dasguptaList = []
        startTime = time.time()
        for epoch in range(epochs):
            for i in range(N):
                randNodeName = np.random.choice(nodeNames[1:], 1)[0]
                randNode = nodeMap[randNodeName]
                randNodeParent = randNode.parent
                config = np.random.choice([0, 1], 1)[0]
                #             print(randNode, randNodeParent)
                # find orientation
                orientation = 0  # the rootnode is on right of its parent
                if not set(randNodeParent.right) == set(
                    (randNode.left + randNode.right)
                ):
                    orientation = 1
                s_array = randNode.left
                t_array = randNode.right
                if orientation == 1:
                    u_array = randNodeParent.right
                else:
                    u_array = randNodeParent.left
                if u_array == randNode.left + randNode.right:
                    print("Picked wrong node to swap!")
                    exit()

                ld_ratio = 1.0
                u_intersect = 0
                d_intersect = 0
                p_ru1 = 0.0
                p_rd1 = 0.0
                u_d1 = 0.0
                u_d2 = 0.0
                exp_ld_ratio = 0.0
                exp_ds_ratio = 0.0
                if config == 0:
                    # new ones
                    u_intersect = UTILS_SIM.findIntersection(
                        s_array, t_array + u_array, duplicateEdgesSet
                    )
                    p_ru1 = (u_intersect * 1.0) / (
                        len(s_array) * (len(t_array) + len(u_array))
                    )
                    p_ru1 = self._sanitize(p_ru1)
                    d_intersect = UTILS_SIM.findIntersection(
                        t_array, u_array, duplicateEdgesSet
                    )
                    p_rd1 = (d_intersect * 1.0) / (len(t_array) * len(u_array))
                    p_rd1 = self._sanitize(p_rd1)
                    # old ones
                    _pr_u = randNodeParent._pr
                    _pr_d = randNode._pr
                    _intersect_u = randNodeParent._interEdges
                    _intersect_d = randNode._interEdges

                    lognum_u = u_intersect * math.log(p_ru1) + (
                        len(s_array) * (len(t_array) + len(u_array)) - u_intersect
                    ) * math.log(1 - p_ru1)
                    lognum_d = d_intersect * math.log(p_rd1) + (
                        len(t_array) * len(u_array) - d_intersect
                    ) * math.log(1 - p_rd1)
                    logdenom_u = _intersect_u * math.log(_pr_u) + (
                        randNodeParent.leftlen * randNodeParent.rightlen - _intersect_u
                    ) * math.log(1 - _pr_u)
                    logdenom_d = _intersect_d * math.log(_pr_d) + (
                        randNode.leftlen * randNode.rightlen - _intersect_d
                    ) * math.log(1 - _pr_d)

                    ld_ratio = (
                        1.0 + (lognum_u + lognum_d - logdenom_u - logdenom_d) / LD_old
                    )

                    """
                    dissimilarity
                    """
                    u_d1 = UTILS_SIM.dissimilarityAvgDist(
                        s_array, t_array + u_array, similarities
                    )
                    u_d2 = UTILS_SIM.dissimilarityAvgDist(
                        t_array, u_array, similarities
                    )
                    d_d2 = randNodeParent._dissim / (
                        (len(randNodeParent.left) + len(randNodeParent.right))
                        * len(randNodeParent.left)
                        * len(randNodeParent.right)
                    )
                    d_d1 = randNode._dissim / (
                        (len(randNode.left) + len(randNode.right))
                        * len(randNode.left)
                        * len(randNode.right)
                    )
                    logds_diff = u_d1 + u_d2 - d_d1 - d_d2
                    d_d1_das = randNode._dissim
                    d_d2_das = randNodeParent._dissim
                    u_d1_das = (
                        (len(s_array) + len(t_array + u_array))
                        * len(s_array)
                        * len(t_array + u_array)
                        * u_d1
                    )
                    u_d2_das = (
                        (len(t_array) + len(u_array))
                        * len(t_array)
                        * len(u_array)
                        * u_d2
                    )

                    logdasgupta_diff = u_d1_das + u_d2_das - d_d1_das - d_d2_das

                    exp_ds_ratio = 1.0
                    if logdasgupta_diff < 100:
                        try:
                            exp_ds_ratio = math.exp(logdasgupta_diff)
                        except:
                            print(logdasgupta_diff)
                            exit()

                if config == 1:
                    # new ones
                    u_intersect = UTILS_SIM.findIntersection(
                        t_array, s_array + u_array, duplicateEdgesSet
                    )
                    p_ru1 = (u_intersect * 1.0) / (
                        len(t_array) * (len(s_array) + len(u_array))
                    )
                    p_ru1 = self._sanitize(p_ru1)
                    d_intersect = UTILS_SIM.findIntersection(
                        s_array, u_array, duplicateEdgesSet
                    )
                    p_rd1 = (d_intersect * 1.0) / (len(s_array) * len(u_array))
                    p_rd1 = self._sanitize(p_rd1)
                    # old ones
                    _pr_u = randNodeParent._pr
                    _pr_d = randNode._pr
                    _intersect_u = randNodeParent._interEdges
                    _intersect_d = randNode._interEdges

                    lognum_u = u_intersect * math.log(p_ru1) + (
                        len(t_array) * (len(s_array) + len(u_array)) - u_intersect
                    ) * math.log(1 - p_ru1)
                    lognum_d = d_intersect * math.log(p_rd1) + (
                        len(s_array) * len(u_array) - d_intersect
                    ) * math.log(1 - p_rd1)
                    logdenom_u = _intersect_u * math.log(_pr_u) + (
                        randNodeParent.leftlen * randNodeParent.rightlen - _intersect_u
                    ) * math.log(1 - _pr_u)
                    logdenom_d = _intersect_d * math.log(_pr_d) + (
                        randNode.leftlen * randNode.rightlen - _intersect_d
                    ) * math.log(1 - _pr_d)

                    ld_ratio = (
                        1.0 + (lognum_u + lognum_d - logdenom_u - logdenom_d) / LD_old
                    )

                    """
                    dissimilarity
                    """
                    u_d1 = UTILS_SIM.dissimilarityAvgDist(
                        t_array, s_array + u_array, similarities
                    )
                    u_d2 = UTILS_SIM.dissimilarityAvgDist(
                        s_array, u_array, similarities
                    )
                    d_d2 = randNodeParent._dissim / (
                        (len(randNodeParent.left) + len(randNodeParent.right))
                        * len(randNodeParent.left)
                        * len(randNodeParent.right)
                    )
                    d_d1 = randNode._dissim / (
                        (len(randNode.left) + len(randNode.right))
                        * len(randNode.left)
                        * len(randNode.right)
                    )
                    logds_diff = u_d1 + u_d2 - d_d1 - d_d2
                    d_d1_das = randNode._dissim
                    d_d2_das = randNodeParent._dissim
                    u_d1_das = (
                        (len(t_array) + len(s_array + u_array))
                        * len(t_array)
                        * len(s_array + u_array)
                        * u_d1
                    )
                    u_d2_das = (
                        (len(s_array) + len(u_array))
                        * len(s_array)
                        * len(u_array)
                        * u_d2
                    )

                    logdasgupta_diff = u_d1_das + u_d2_das - d_d1_das - d_d2_das

                    exp_ds_ratio = 1.0
                    if logdasgupta_diff < 100:
                        try:
                            exp_ds_ratio = math.exp(logdasgupta_diff)
                        except:
                            print(logdasgupta_diff)
                            exit()

                prob = exp_ds_ratio

                LD_new = LD_old
                logds_new = logds_old
                logdasgupta_new = logdasgupta_old

                # take a step with probability ld_ratio
                step = (
                    True if prob > 0.0 and np.random.uniform() < min(1, prob) else False
                )
                if step:
                    LD_new = ld_ratio * LD_old
                    logds_new = logds_diff + logds_old
                    logdasgupta_new = logdasgupta_diff + logdasgupta_old

                    self._treeObj.updateTree(
                        config,
                        randNode,
                        randNodeParent,
                        u_intersect,
                        d_intersect,
                        p_ru1,
                        p_rd1,
                        u_d1_das,
                        u_d2_das,
                        orientation,
                    )
                    # update LD
                    LD_old = LD_new
                    logds_old = logds_new
                    logdasgupta_old = logdasgupta_new

                ldList.append(LD_new)
                dsList.append(logds_new)
                dasguptaList.append(logdasgupta_new)
            if epoch % (max(1, epochs / 100)) == 0:
                print(
                    "Epoch: {:5d}, logL: {}, time: {:6.4f}".format(
                        epoch, LD_old, time.time() - startTime
                    )
                )
        return (ldList, dsList, dasguptaList)
