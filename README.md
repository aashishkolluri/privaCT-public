# PrivaCT: **Priva**te **C**lustering **T**rees for Federated Networks

*PrivaCT enables hierarchically clustering the users of an undirected and unattributed federated social network with local differential privacy.*

### Wait... what is a federated social network?

In a federated social netowrk, users keep their personal information such as contacts on their end devices. All users register with the social network by giving their only details such as phone number or email address. Using this information, the central authority controlling the federated social network is responsible for connecting the users to each other while not being able to know any other personal information about the users. Does this remind you of any such real world social network? (Hint: [Signal](https://support.signal.org/hc/en-us/articles/360007061452-Does-Signal-send-my-number-to-my-contacts-]))

### Ok, then why clustering in a federated network?

Social networks capture rich information that can be used for personalized services such as online advertising and recommendations. One of the most important queries to power these applications is to understand the community structure around every user. The reason is that users tend to connect, accept and behave like other users that are in their community (neighbors, second-degree neighbors, etc.). This query is trivial to do in the centralized setup when the whole network is available to the authority. However, many privacy-sensitive users are moving towards federated services. So, how can one provide personalized services to users in the federated setup while preserving their privacy? Essentially, how to query the community around the user at various granularities in the federated setup? PrivaCT does just that.

### Why is this useful?

Today any federated network (say, Signal) can use PrivaCT to cluster their users hierarchically and provide this as a valuable piece of information to a third-party service provider such as Spotify/Netflix to help with their recommendations.

### What do users get?

Users get a strong privacy guarantee in terms of local differential privacy while also getting personalized services from their favorite third-party service providers.

### Is this repository ready for production?

No, this project is just a prototype and is far from production level.

### What are its limitations?

The current algorithm is not parallelizable and does not scale for large networks (say, > 100,000 users) even if the code is written in C++.

## Installation
We recommend using [Anaconda](https://www.anaconda.com) as it is easy to manage the environment and packages.

Create a virtual environment using anaconda.
```
conda create -n venv python==3.8
conda activate venv
```
Add this channel for packages
```
conda config --add channels conda-forge
```

Install the following packages
```
conda install -y numpy
conda install -y networkx
conda install -y anytree
conda install -y dill
conda install -y scikit-learn
pip install python-louvain
```

## Learning a hierarchical cluster tree (HCT)

PrivaCT just needs an undirected network as the input with some parameters.

### Usage:

First, set the data home (`DATA_HOME`) in env.py to the root data folder.

```
python  main.py --dataset [DName] --dfPath [EFileName] --createHCT --all --epochs [Epochs] --seed [Seed]
or
python  main.py --dataset [DName] --dfPath [EFileName] --createHCT --eps_l [Epsilon] --epochs [Epochs] --seed [Seed]
```
* `DName` - dataset name in data folder (e.g., `lastfm-small`, `delicious`)
* `EFileName` - edges file name in the `DName` folder (e.g., `edges_final.txt`)
* `Epochs` - the number of epochs you want to learn a HCT for (default: 1000)
* `Epsilon` - the privacy budget (default: 1.0) which is actually 2.0 in LDP
* `Seed` - the random seed to fix for generating HCT (default: 1234)

### Output:

The outputs will be pickled trees stored in `DATADIR/DName/Seed/`

To load the trees
```
> from treefunc import *
> hct = Tree(0, [TreeFilePath], [Seed])
> root = hct.treeRoot
> root.left = list of users in left cluster at root
> root.children = [leftChild, rightChild]
```
Check out this awesome library for more tree manuipulations [anytree](https://anytree.readthedocs.io/en/latest/).

## Using the HCT for social recommendations

PrivaCTCF is a collaborative filtering algorithm based on PrivaCT trees. For cold-start users,

### Usage:
The preprocessed data is given for one dataset in lastfm. First, set the data home (`DATA_HOME`) and `RESULT_DIR` in `env.py` to the root data folder.

```
python recommender.py --dataset [DName] --method privaCTCF --seedSplit [SeedSplit] --seed [Seed]
```
* `DName` - dataset name in data folder (say lastfm-small, delicious)
* `Seed` - the random seed to fix for generating HCT (default: 1234)
* `SeedSplit` - fix seed for test train split

### Output:
NDCG, MAP scores.

### Custom Preprocessing:
Any dataset has to be first preprocessed before running PrivaCTCF on it. It need two important files:
* `ratings.txt` contains tuples `(userID, itemID, rating)`
* `trusts.txt` contains tuples `(userID, friendID)`

First the largest connected component (`G'`) of the network (`G`) is taken. Finally, the users are named (`userID`) from `0...len(G')-1` deterministically, see `data/lastfm/preprocessing.py`.

The `ratings.txt` file has `userID` named as mentioned above. The itemID also belongs to range `0...M-1` and the names are chosen by the order in which the items appear in the original items file.

The `trusts.txt` file similarly has the `userID` and `friendID`.

* `M` - total number of items

# Contribute

We may not find time to maintain this repository. In case you have any questions then please feel free to reach out one of us.

Aashish Kolluri

Mail: e0321280@u.nus.edu

Teodora Baluta

Mail: teodorab@comp.nus.edu.sg

Prateek Saxena

Mail: prateeks@comp.nus.edu.sg
