import numpy as np
import matplotlib

matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
})

import matplotlib.pyplot as plt
import networkx as nx
import dill as pickle
import os
import sys
from anytree import Node, RenderTree
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dpAlg'))

from treefunc import *

DATA_PATH = '/home/aashish/privact-data'
RESULTS = './figures/'

if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)

# print and compute some graph statistics
def graph_stats(title, g):
    print("Simple stats for: " + title)
    nodes = len(g.nodes())
    edges = len(g.edges())
    density = (len(g.edges())*2.0)/(len(g.nodes())*(len(g.nodes())-1))
    print("Nodes: {:10f}".format(nodes))
    print("Edges: {:10f}".format(edges))
    print("Density: {:10.4f}".format(density))
    return nodes, edges, density


# load graph returning the largest connected component
def load_graph(dataset_dir, dataset, dfPath, dataFormat):
    # Find the data file path containing graph edges
    datafilePath = os.path.join(dataset_dir, dfPath)
    # Read the graph from the file depending on the format
    graph = {}
    if dataFormat == 'txt':
        graph = nx.read_edgelist(datafilePath)
    elif dataFormat == 'gml':
        graph = nx.read_gml(datafilePath)
    else:
        print('Unknown data format for graph file. Exiting!')
        exit()

    # Select the largest connected component if it is not connected
    cg = None
    cg = max(nx.connected_components(graph), key=len)
    cg = graph.subgraph(cg).copy()

    # Convert all the node names and edges to integral values starting from 0
    cg = nx.convert_node_labels_to_integers(cg)
    N = len(cg.nodes())
    E = len(cg.edges())

    return cg


def load_loss_file(datadir, dataset, seed, is_global=False):
    '''
        Load losses file
    '''
    if is_global:
        filename = os.path.join(datadir, dataset, str(seed),
                                'loss_global_all_epochs_1000.pkl')
    else:
        filename = os.path.join(datadir, dataset, str(seed),
                                'loss_all_epochs_1000.pkl')

    print('Loading from {}; is_global={}'.format(filename, is_global))
    if not os.path.exists(filename):
        return None
    loss_dict = {}
    with open(filename, 'rb') as fp:
        loss_dict = pickle.load(fp)
    return loss_dict

def load_dataset_MCMC(dataset, seed, results_dir):
    results_dict[dataset]['mcmc_losses_dict'] = {}
    results_dict[dataset]['global_mcmc_losses_dict'] = {}
    for seed in seeds:
        results_dict[dataset]['mcmc_losses_dict'][seed] = \
            load_loss_file(DATA_PATH, dataset, seed)
        results_dict[dataset]['global_mcmc_losses_dict'] = \
            load_loss_file(DATA_PATH, dataset, seed)


# Get the losses of the mcmc process
def get_MCMC_loss(cg, eps, loss_dict):
    N = len(cg.nodes())
    if eps not in loss_dict:
        print('!! Missing eps={}'.format(eps))
        return []
    ldList, dsList, dasList = loss_dict[eps]
    normalizedIDList = []
    normalizedDSList = []
    normalizedDASList = []
    for i in range(len(ldList)):
        if not i%N==0:
            continue
        normalizedIDList.append(sum(ldList[i:i+N])/N)
        normalizedDSList.append(sum(dsList[i:i+N])/N)
        normalizedDASList.append(sum(dasList[i:i+N])/N)

    return normalizedDASList

# Get the average losses of the mcmc process
def get_MCMC_avg_loss(cg, seeds, loss_dict, loss_type='ID'):
    N = len(cg.nodes())
    epochs = 1000
    ldList = np.zeros(N*epochs)
    dsList = np.zeros(N*epochs)
    dasList = np.zeros(N*epochs)
    num_seeds = 0

    for seed in seeds:
        if seed not in loss_dict or not loss_dict[seed]:
            print('NO SEEDDDD {}'.format(seed))
            continue
        ldListSeed, dsListSeed, dasListSeed = loss_dict[seed]
        ldList += np.array(ldListSeed)
        #dsList += np.array(dsListSeed)
        #dasList += np.array(dasListSeed)
        num_seeds += 1

    if num_seeds == 0:
        print('No seeds found, cannot average')
        return []

    ldList /= num_seeds
    dsList /= num_seeds
    dasList /= num_seeds
    normalizedIDList = []
    normalizedDSList = []
    normalizedDASList = []
    for i in range(len(ldList)):
        if not i%N==0:
            continue
        normalizedIDList.append(sum(ldList[i:i+N])/N)
        normalizedDSList.append(sum(dsList[i:i+N])/N)
        normalizedDASList.append(sum(dasList[i:i+N])/N)

    if loss_type == 'ID':
        return normalizedIDList
    if loss_type == 'Dasgupta':
        return normalizedDASList
    if loss_type == 'DS':
        return normalizedDSList

def plot_MCMC_losses(dataset, all_eps, seeds, cg, local_loss_dict,
                     global_loss_dict, display=False, save_file_prefix='mle_cost',
                     local_legend_prefix='LDP',
                     global_legend_prefix='GDP'):
    print('*'*80)
    print(dataset)
    legend_eps = []
    print(local_loss_dict.keys())
    for eps in all_eps:
        if eps not in local_loss_dict:
            print('!! Missing eps={}'.format(eps))
            return []

        normalizedIDList = get_MCMC_avg_loss(cg, seeds, local_loss_dict[eps],
                                             loss_type='ID')
        plt.plot(list(range(len(normalizedIDList))),normalizedIDList)

        if eps not in global_loss_dict:
            print('!! Missing eps={}'.format(eps))
            return []

        globalNormalizedIDList = get_MCMC_avg_loss(cg, seeds, global_loss_dict[eps],
                                                   loss_type='ID')

        plt.plot(list(range(len(normalizedIDList))), globalNormalizedIDList)

        plt.xlabel('steps/n', fontsize=16)
        plt.ylabel('$\log{(C_M)}$', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.plot(list(range(len(normalizedDSList))),normalizedDSList)
        # plt.plot(list(range(len(normalizedDASList))),normalizedDASList)
        legend_eps.append('LDP-$\epsilon$-{}'.format(eps))
        legend_eps.append('GDP-$\epsilon$-{}'.format(eps))

    print('*'*80)

    plt.legend(legend_eps, loc='upper right', prop={'size': 12})
    plt.savefig(os.path.join(RESULTS, '{}-mle_cost.pdf'.format(dataset)),
                format='pdf', bbox_inches='tight')
    if display:
        plt.show()
    plt.clf()

def computeDasguptaCost(N, sim_file, fileTreePath, seed):
    with open(sim_file, 'rb') as fp:
        degree_vec = pickle.load(fp)
        similarities = pickle.load(fp)

    # Create a Tree Object or retrieve it from fle
    hrgTree = Tree(N, fileTreePath, seed)

    # Reverse the nodemap dictionary
    nodeMapRev = hrgTree.reverseNodeNames()

    # Do a bfs to get the internal nodes at each level
    bfsList, nodesAtLevel, maxdepth = hrgTree.bfs()

    scoreOPT=0.0
    scoreClique=0.0
    for name in hrgTree.nodeNames:
        node = hrgTree.nodeMap[name]
        lista = node.left
        listb = node.right
        cartProd = [(a, b) for a in lista for b in listb]
        for each in cartProd:
            dissim = similarities[each[0]][each[1]]
            scoreOPT+=dissim*(len(node.left)+len(node.right))
            scoreClique += len(node.left)+len(node.right)

    return scoreOPT, scoreClique

def dasguptaCosts(N, fileTreeDir, seed, eps, epochs):

    fileTreePath = os.path.join(fileTreeDir, 'dp' + '_eps_l_' + str(eps) + \
                                '_epochs_' + str(epochs) + '.pkl')

    sim_file = os.path.join(fileTreeDir, 'dp' + '_eps_l_' + str(0.0)+'_seed_' + \
                            str(seed) + '_epochs_' + str(epochs) + '.sim')

    fileTreePathOPT = os.path.join(fileTreeDir, 'dp' + '_eps_l_' + str(0.0) + \
                                   '_epochs_' + str(epochs) + '.pkl')

    scoreDP, scoreClique = computeDasguptaCost(N, sim_file, fileTreePath, seed)
    scoreOPT, scoreClique = computeDasguptaCost(N, sim_file, fileTreePathOPT, seed)
    print('Optimal score:{}, Optimal DP score:{},'
          'Optimal clique score:{},'
          'utility loss:{}'.format(scoreOPT, scoreDP, scoreClique,
                                   abs(scoreOPT-scoreDP)/scoreOPT))
    return scoreClique, scoreDP, scoreOPT


def computeAllDasguptaScores(datasets, seeds, all_eps, dasguptaScores):
    df_path = 'edges_final.txt'
    data_format = 'txt'
    epochs = 1000

    for dataset in datasets:
        dataset_dir = os.path.join(DATA_PATH, dataset)
        cg = load_graph(dataset_dir, dataset, df_path, data_format)
        nodes, edges, density = graph_stats(dataset, cg)

        dasguptaScores[dataset] = {}

        for eps in all_eps:
            avg_scoreClique = 0
            avg_scoreDP = 0
            avg_scoreOPT = 0
            num_seeds = 0
            avg_utility_loss = 0
            avg_base = 0
            for seed in seeds:
                DATADIR = os.path.join(DATA_PATH, dataset)

                fileTreeDir = os.path.join(DATADIR, str(seed))

                N = len(cg.nodes())
                scoreClique, scoreDP, scoreOPT = dasguptaCosts(N, fileTreeDir, seed, eps, epochs)
                avg_utility_loss += abs(scoreOPT-scoreDP)/scoreOPT
                avg_base += scoreOPT / scoreClique

                num_seeds += 1
                avg_scoreClique += scoreClique
                avg_scoreDP += scoreDP
                avg_scoreOPT += scoreOPT

            avg_scoreClique /= num_seeds
            avg_scoreDP /= num_seeds
            avg_scoreOPT /= num_seeds
            avg_utility_loss /= num_seeds
            avg_base /= num_seeds

            dasguptaScores[dataset][eps] = [nodes, avg_scoreClique, avg_scoreDP, avg_scoreOPT,
                                            avg_utility_loss]
            print('{} - eps={}: avg utility loss {}'.format(dataset, eps, avg_utility_loss))
            print('{} - eps={}, scoreOPT/scoreClique {}'.format(dataset, eps, avg_base))

def computeThDasgupta(datasets, all_eps, dasgupta_scores):
    for dataset in datasets:
        num_nodes = dasgupta_scores[dataset][0.25][0]
        scoreClique = dasgupta_scores[dataset][0.25][1]
        scoreOPT = dasgupta_scores[dataset][0.25][3]

        k = int(math.log(num_nodes))

        for eps in all_eps:
            th_bound = 4*k/eps*(eps + 3.0 / math.sqrt(k)) * scoreClique / scoreOPT

            print('{}- eps={}: th bound {}'.format(dataset, eps, th_bound))


def save_to_csv(dataset, all_eps, seeds, cg, local_loss_dict, global_loss_dict,
                display=False):
    nodot_eps = {0.5: '05', 1.0: '10', 2.0:'20'}
    for eps in all_eps:
        if eps not in local_loss_dict:
            print('!! Missing eps={}'.format(eps))
            return []

        normalizedIDList = get_MCMC_avg_loss(cg, seeds, local_loss_dict[eps],
                                             loss_type='ID')

        with open('{}_{}_mcmc_losses.csv'.format(dataset, nodot_eps[eps]), 'w') as f:
            loss_writer = csv.writer(f, delimiter=',')
            loss_writer.writerow(['steps/n', 'loss'])

            for i in range(len(normalizedIDList)):
                loss_writer.writerow([i, normalizedIDList[i]])

        if eps not in global_loss_dict:
            print('!! Missing eps={}'.format(eps))
            return []

        globalNormalizedIDList = get_MCMC_avg_loss(cg, seeds, global_loss_dict[eps],
                                                   loss_type='ID')

        with open('global_{}_{}_mcmc_losses.csv'.format(dataset, nodot_eps[eps]), 'w') as f:
            loss_writer = csv.writer(f, delimiter=',')
            loss_writer.writerow(['steps/n', 'loss'])

            for i in range(len(globalNormalizedIDList)):
                loss_writer.writerow([i, globalNormalizedIDList[i]])


def main():
    datasets = ['lastfm-small', 'douban', 'delicious']
    data_format = 'txt'
    df_path = 'edges_final.txt'
    seeds = [1234, 144, 1766]
    epochs = 1000

    # this is so broken rn
    all_eps = [0.5, 1.0, 2.0]
    all_local_eps = {0.25: 0.5, 0.5: 1.0, 1.0: 2.0}

    results_dict = {}
    dasgupta_scores = {}

    if len(sys.argv) > 1 and sys.argv[1] == 'mcmc':
        for dataset in datasets:
            dataset_dir = os.path.join(DATA_PATH, dataset)
            cg = load_graph(dataset_dir, dataset, df_path, data_format)
            nodes, edges, density = graph_stats(dataset, cg)

            local_mcmc_loss_dict = {}
            global_mcmc_loss_dict = {}
            for seed in seeds:
                local_mcmc_loss_dict[seed] = load_loss_file(DATA_PATH, dataset,
                                                            seed)
                global_mcmc_loss_dict[seed] = load_loss_file(DATA_PATH, dataset,
                                                             seed, is_global=True)

            local_loss_dict = {}
            global_loss_dict = {}
            for eps in all_eps:
                global_loss_dict[eps] = {}
            for eps in all_local_eps.keys():
                local_loss_dict[all_local_eps[eps]] = {}

            for seed in seeds:
                if local_mcmc_loss_dict[seed] == None:
                    print('{} - MCMC LOSS - Missing local seed={}'.format(dataset, seed))
                    continue
                if global_mcmc_loss_dict[seed] == None:
                    print('{} - MCMC LOSS - Missing global seed={}'.format(dataset, seed))
                    continue
                for eps in all_eps:
                    if eps not in global_mcmc_loss_dict[seed]:
                        print('{} - MCMC LOSS - Missing local eps={}'.format(dataset, eps))
                        continue
                    global_loss_dict[eps][seed] = global_mcmc_loss_dict[seed][eps]

                for eps in all_local_eps.keys():
                    if eps not in local_mcmc_loss_dict[seed]:
                        print('{} - MCMC LOSS - Missing local eps={}'.format(dataset, eps))
                        continue
                    local_loss_dict[all_local_eps[eps]][seed] = local_mcmc_loss_dict[seed][eps]

            save_to_csv(dataset, all_eps, seeds, cg, local_loss_dict, global_loss_dict,
                        display=False)
            plot_MCMC_losses(dataset, all_eps, seeds, cg, local_loss_dict,
                             global_loss_dict, display=False)

    else:
        computeAllDasguptaScores(datasets, seeds, all_local_eps.keys(), dasgupta_scores)
        computeThDasgupta(datasets, all_eps, dasgupta_scores)

if __name__ == '__main__':
    main()
