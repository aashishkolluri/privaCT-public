import networkx as nx


# print some graph statistics
def print_graph_stats(title, g):
    print("Simple stats for: " + title)
    print("Nodes: {:10f}".format(len(g.nodes())))
    print("Edges: {:10f}".format(len(g.edges())))
    print(
        "Density: {:10.4f}".format(
            (len(g.edges()) * 2.0) / (len(g.nodes()) * (len(g.nodes()) - 1))
        )
    )

if __name__=='__main__':

    # First load the graph
    graph = nx.read_edgelist('./edges_final.txt')
    # Select the largest connected component if it is not connected
    cg = graph
    if not nx.is_connected(graph):
        print('The graph is not connected')
        cg = max(nx.connected_components(graph), key=len)
        cg = graph.subgraph(cg).copy()

    print_graph_stats('LastFM', cg)

    nodeMap = {}
    revNodeMap = {}
    # create new labels
    count = 0
    for node in cg.nodes():
        nodeMap[node] = count
        revNodeMap[count] = node
        count += 1
    # assign the new labels
    cg = nx.relabel_nodes(cg, nodeMap)

    itemMap = {}
    count = 0
    # Now load the user-artists and replace the user with its new name
    with open('ratings.txt', 'w') as fp1:
        with open('user_artists.dat', 'r') as fp:
            lines = fp.readlines()
            for line in lines[1:]:
                items = line.split()
                if items[0] in nodeMap:
                    if items[1] in itemMap:
                        fp1.write(str(nodeMap[items[0]])+'\t'+str(itemMap[items[1]])+'\t'+ items[2]+'\n')
                    else:
                        itemMap[items[1]] = count
                        count+=1
                        fp1.write(str(nodeMap[items[0]])+'\t'+str(itemMap[items[1]])+'\t'+ items[2]+'\n')
                else:
                    print(f"Node {items[0]} not in the connected graph. Removing its ratings")

    # Now load the user-artists and replace the user with its new name
    with open('trusts.txt', 'w') as fp1:
        with open('edges_final.txt', 'r') as fp:
            lines = fp.readlines()
            for line in lines[1:]:
                items = line.split()
                if (items[0] in nodeMap) and (items[1] in nodeMap):
                    fp1.write(str(nodeMap[items[0]])+'\t'+str(nodeMap[items[1]])+'\n')
                    fp1.write(str(nodeMap[items[1]])+'\t'+str(nodeMap[items[0]])+'\n')
                else:
                    print(f"Node {items[0]} or {items[1]} not in the connected graph. Removing its edge")