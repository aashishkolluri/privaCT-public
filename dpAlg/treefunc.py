from anytree import Node, RenderTree
from collections import defaultdict
from collections import deque
import os
import dill as pickle
import math
import numpy as np

# This holds the functions needed to operate on the tree
class Tree:
    def __init__(self, N, filepath, seed, prng=np.random):
        # Define tree root
        self.treeRoot = Node(N)
        self.seed = seed
        self.prng = prng
        # contains all the nodeNames
        self.nodeNames = []
        # contains the map nodeName:Node(pointer to node in the tree)
        self.nodeMap = {}
        self.nodeMapRev = {}
        self.nodeNames.append("i_" + str(N))
        self.nodeMap["i_" + str(N)] = self.treeRoot
        if os.path.isfile(filepath):
            with open(filepath, "rb") as fp:
                self.treeRoot = pickle.load(fp, pickle.HIGHEST_PROTOCOL)
                self.nodeNames = pickle.load(fp, pickle.HIGHEST_PROTOCOL)
                self.nodeMap["i_" + str(N)] = self.treeRoot
                self.loadMaps(self.treeRoot)
                self.reverseNodeNames()
                self.seed = pickle.load(fp, pickle.HIGHEST_PROTOCOL)
                print(
                    "Started with likelihood: {}".format(
                        self.getLikelihood(self.treeRoot)
                    )
                )

    # construct the nodemaps from the tree
    def loadMaps(self, root):
        # The stop condition of DFS recursion, when it reaches an internal node with two leaves
        if len(root.left + root.right) == 2:
            self.nodeMap["i_" + str(root.name)] = root
            return
        self.nodeMap["i_" + str(root.name)] = root
        if not root.leftlen == 1:
            self.loadMaps(root.children[0])
        if not root.rightlen == 1:
            self.loadMaps(root.children[1])

    def getLikelihood(self, root):
        """
           Call to get the log likelihood of the tree
        """
        ld = 0.0
        if not root.leftlen == 1:
            ld += self.getLikelihood(root.children[0])
        if not root.rightlen == 1:
            ld += self.getLikelihood(root.children[1])
        e = 0.0
        try:
            e = root._interEdges
        except:
            print(root)
            exit()
        n_l = root.leftlen
        n_r = root.rightlen
        p_r = (e * 1.0) / (n_l * n_r)

        if p_r == 0.0:
            p_r = 0.000000001
        if p_r == 1.0:
            p_r = 0.999999999
        return ld + e * math.log(p_r) + (n_l * n_r - e) * math.log(1 - p_r)

    # Construct a random tree with input being rootnode, list of vertices
    def randomTree(self, root, lists, count, nodeNames, nodeMap):
        """
            nodeNames: It is a list of names of internal nodes of the tree
            nodeMap: {nodeName: pointer to the actual node}
            The above two datastructures are needed for dealing with AnyTree library
        """

        # The stop condition of DFS recursion, when it reaches an internal node with two leaves
        if len(lists) == 2:
            root.left = [lists[0]]
            root.leftlen = 1
            root.right = [lists[1]]
            root.rightlen = 1
            node1 = Node(lists[0], parent=root)
            node2 = Node(lists[1], parent=root)
            return count

        tempArray = list(range(1, len(lists) - 1))
        pivot = self.prng.choice(tempArray, 1)[0]
        leftList = lists[0:pivot]
        rightList = lists[pivot : len(lists)]
        root.left = leftList
        root.leftlen = len(leftList)
        root.right = rightList
        root.rightlen = len(rightList)
        retcount = count
        # when the internal node has just one node in its left subtree
        if len(leftList) == 1:
            node1 = Node(leftList[0], parent=root)
        else:
            node1 = Node(count, parent=root)
            name = "i_" + str(count)
            nodeNames.append(name)
            nodeMap[name] = node1
            retcount = self.randomTree(node1, leftList, count + 1, nodeNames, nodeMap)
        # when the internal node has just one node in its right subtree
        if len(rightList) == 1:
            node1 = Node(rightList[0], parent=root)
        else:
            node1 = Node(retcount, parent=root)
            name = "i_" + str(retcount)
            nodeNames.append(name)
            nodeMap[name] = node1
            retcount = self.randomTree(
                node1, rightList, retcount + 1, nodeNames, nodeMap
            )
        return retcount

    def assignLabel(self):
        for _, node in self.nodeMap.items():
            node.label = min([int(x.name) for x in node.leaves])
        for leaf in self.treeRoot.leaves:
            leaf.label = int(leaf.name)
        return

    def makeOrPTree(self):
        for _, node in self.nodeMap.items():
            if node.children[0].label > node.label:
                s_array = node.left
                t_array = node.right
                s_node = node.children[0]
                t_node = node.children[1]
                node.children = []
                node.children = [t_node, s_node]
                node.left = t_array
                node.right = s_array
                node.leftlen = len(t_array)
                node.rightlen = len(s_array)
        return

    # A function to update the tree during the metropolis-hastings based algorithm
    def updateTree(
        self,
        config,
        randNode,
        randNodeParent,
        u_intersect,
        d_intersect,
        p_ru1,
        p_rd1,
        u_d1,
        u_d2,
        orientation,
    ):

        s_array = randNode.left
        t_array = randNode.right

        s_node = randNode.children[0]
        t_node = randNode.children[1]
        # find the u_node
        u_id = 0
        u_array = randNodeParent.left
        if orientation == 1:
            u_id = 1
            u_array = randNodeParent.right
            if len(randNodeParent.right) > 1:
                try:
                    if not set(randNodeParent.right) == set(
                        randNodeParent.children[1].left
                        + randNodeParent.children[1].right
                    ):
                        print("Did a mistake in picking the node to swap!")
                        exit()
                except:
                    print(randNode, randNodeParent, randNodeParent.children[1])
        u_node = randNodeParent.children[u_id]

        if config == 0:
            # updating all the above for parent node
            if orientation == 1:
                # updating pointers for the selected node
                randNodeParent.children = []  # detach u randNode
                randNode.children = []  # detach s t
                randNode.children = [t_node, u_node]
                randNode.left = t_array
                randNode.right = u_array
                # updating lengths
                randNode.leftlen = len(t_array)
                randNode.rightlen = len(u_array)
                # updating intersection and p_r
                randNode._interEdges = d_intersect
                randNodeParent._interEdges = u_intersect
                randNode._pr = p_rd1
                randNodeParent._pr = p_ru1
                randNode._dissim = u_d2
                randNodeParent._dissim = u_d1
                randNodeParent.children = [s_node, randNode]
                randNodeParent.left = s_array
                randNodeParent.right = t_array + u_array
                randNodeParent.leftlen = len(s_array)
                randNodeParent.rightlen = len(t_array) + len(u_array)
                # enforce order property
                node = randNode
                if node.children[0].label > node.children[1].label:
                    s_array_1 = node.left
                    t_array_1 = node.right
                    s_node_1 = node.children[0]
                    t_node_1 = node.children[1]
                    node.children = []
                    node.children = [t_node_1, s_node_1]
                    node.left = t_array_1
                    node.right = s_array_1
                    node.leftlen = len(t_array_1)
                    node.rightlen = len(s_array_1)
                node.label = node.children[0].label
                if not randNode.label == min([int(x) for x in randNode.left]):
                    print("Here0", randNodeParent, randNode, randNode.children[0])
                    exit()

            else:
                # updating pointers for the selected node
                randNodeParent.children = []  # detach u randNode
                randNode.children = []  # detach s t
                randNode.children = [u_node, t_node]
                randNode.left = u_array
                randNode.right = t_array
                # updating lengths
                randNode.leftlen = len(u_array)
                randNode.rightlen = len(t_array)
                # updating intersection and p_r
                randNode._interEdges = d_intersect
                randNodeParent._interEdges = u_intersect
                randNode._pr = p_rd1
                randNodeParent._pr = p_ru1
                randNode._dissim = u_d2
                randNodeParent._dissim = u_d1
                randNodeParent.children = [randNode, s_node]
                randNodeParent.right = s_array
                randNodeParent.left = u_array + t_array
                randNodeParent.rightlen = len(s_array)
                randNodeParent.leftlen = len(u_array) + len(t_array)
                randNode.label = randNode.children[0].label
                if not randNode.label == min([int(x) for x in randNode.left]):
                    print("Here1", randNode, randNode.children[0])
                    exit()

        elif config == 1:
            if orientation == 1:
                # updating pointers
                randNodeParent.children = []  # detach u randNode
                randNode.children = []  # detach t s
                randNode.children = [s_node, u_node]
                randNode.left = s_array
                randNode.right = u_array
                # updating lengths
                randNode.leftlen = len(s_array)
                randNode.rightlen = len(u_array)
                # updating intersection and p_r
                randNode._interEdges = d_intersect
                randNodeParent._interEdges = u_intersect
                randNode._pr = p_rd1
                randNodeParent._pr = p_ru1
                randNode._dissim = u_d2
                randNodeParent._dissim = u_d1
                # updating all the above for parent node
                randNodeParent.children = [randNode, t_node]
                randNodeParent.left = s_array + u_array
                randNodeParent.right = t_array
                randNodeParent.leftlen = len(s_array) + len(u_array)
                randNodeParent.rightlen = len(t_array)
                if not randNode.label == min([int(x) for x in randNode.left]):
                    print("Here2", randNode, randNode.children[0])
                    exit()
            else:
                # updating pointers
                randNodeParent.children = []  # detach u randNode
                randNode.children = []  # detach t s
                randNode.children = [u_node, s_node]
                randNode.left = u_array
                randNode.right = s_array
                # updating lengths
                randNode.leftlen = len(u_array)
                randNode.rightlen = len(s_array)
                # updating intersection and p_r
                randNode._interEdges = d_intersect
                randNodeParent._interEdges = u_intersect
                randNode._pr = p_rd1
                randNodeParent._pr = p_ru1
                randNode._dissim = u_d2
                randNodeParent._dissim = u_d1
                # updating all the above for parent node
                randNodeParent.children = [randNode, t_node]
                randNodeParent.left = u_array + s_array
                randNodeParent.right = t_array
                randNodeParent.leftlen = len(s_array) + len(u_array)
                randNodeParent.rightlen = len(t_array)
                randNode.label = randNode.children[0].label
                if not randNode.label == min([int(x) for x in randNode.left]):
                    print("Here3", randNode, randNode.children[0])
                    exit()
        # catch violations of order property
        if randNode.children[0].label > randNode.label:
            print(randNode, randNode.children[0])
            print("Order property violated!!")
            exit()
        return

    def reverseNodeNames(self):
        for each in self.nodeNames:
            node = self.nodeMap[each]
            self.nodeMapRev[node] = each
        return self.nodeMapRev

    def bfs(self):
        rootnode = self.treeRoot
        nodeMapRev = self.nodeMapRev
        initlevel = 0
        nodeList = []
        nodesAtLevel = defaultdict(list)
        # Append the root node
        nodeList.append(rootnode)
        # Add the level to the dict
        nodesAtLevel[initlevel].append(rootnode)
        # The queue to maintain during bfs
        q = deque()
        q.append((initlevel, rootnode))
        maxlevel = initlevel

        while not len(q) == 0:
            level, node = q.popleft()
            if len(node.children) == 0:
                continue
            children = node.children
            if children[0] in nodeMapRev:
                nodeList.append(children[0])
                nodesAtLevel[level + 1].append(children[0])
                q.append((level + 1, children[0]))
            if children[1] in nodeMapRev:
                nodeList.append(children[1])
                nodesAtLevel[level + 1].append(children[1])
                q.append((level + 1, children[1]))
            maxlevel = max(level, maxlevel)
        return nodeList, nodesAtLevel, maxlevel

    def saveModel(self, filepath):
        print("Saving model to: ", filepath)
        print("Ended with likelihood: {}".format(self.getLikelihood(self.treeRoot)))
        with open(filepath, "wb") as fp:
            pickle.dump(self.treeRoot, fp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.nodeNames, fp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.seed, fp, pickle.HIGHEST_PROTOCOL)
        return
