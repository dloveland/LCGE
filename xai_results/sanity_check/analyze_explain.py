import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

test_ids = [i for i in range(1, 6)]
iter_ids = [i for i in range(0, 5)]
# head_path = "trainedGNN"
head_path = "randomizedGNN"

attr_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
attr_color = {0: 'green', 1: 'red', 2: 'blue', 3: 'cyan', 4: 'magenta', 5: 'white', 6: 'yellow'}


def graph_draw(x, a):
    # graph = nx.Graph()
    graph = nx.from_numpy_matrix(a)
    print('/test_idx_{0}_iter_{1}.png'.format(t_id, iter_id))
    # print(graph)
    attr = {k: v for k, v in zip(np.where(x)[0], np.where(x)[1])}
    # print("attr: ", attr)
    labels = {}
    # color = ''
    color = []
    for n in attr:
        labels[n] = attr_dict[attr[n]]
        color.append(attr_color[attr[n]])

    nx.draw(graph, labels=labels, node_color=color, with_labels=True)
    # plt.show()
    plt.savefig(head_path + '/test_idx_{0}_iter_{1}.png'.format(t_id, iter_id))
    plt.clf()


for t_id in test_ids:
    Xs = []
    As = []
    probs = []
    sizes = []
    atom_cats = []
    random_molecule_id = random.randint(0,10)
    for iter_id in iter_ids:
        path = os.path.join(head_path, 'test_idx_{0}_iter_{1}.npz'.format(t_id, iter_id))
        test = np.load(path)
        X = test['arr_0']
        A = test['arr_1']
        graph_draw(X, A)
        prob = test['arr_2']
        # size = (A == 1).sum()//2
        size = A.shape[0]
        sizes.append(size)
        Xs.append(X)
        As.append(A)
        probs.append(prob)
        atom_cats.append(np.count_nonzero(X, axis=0))

    Xs = np.array(Xs)
    As = np.array(As)
    probs = np.array(probs)

    atom_cats = np.array(atom_cats)
    print("atom cats: ", atom_cats)
    atoms = [attr_dict[i] for i in range(7)]
    df = pd.DataFrame(atom_cats, columns=atoms)
    ax = sns.violinplot(data=df, palette={k: v for k, v in zip(attr_dict.values(), attr_color.values())})
    ax.set_xlabel("Atom type", fontsize=12)
    ax.set_ylabel("# atoms", fontsize=12)
    fig = ax.get_figure()
    fig.savefig(str(head_path) + "/atoms_distribution_test_{0}.png".format(t_id))
    fig.clf()
    #

    # df = pd.DataFrame(sizes)
    # if cls == 0:
    #     ax = sns.violinplot(data=df)
    # else:
    #     ax = sns.violinplot(data=df, color="green")
    # # ax.set_xlabel("Edge Size", fontsize=12)
    # ax.set_ylabel("Node distribution", fontsize=12)
    # fig = ax.get_figure()
    # fig.savefig(directory+"/node_size_distribution.png")
    # fig.clf()

    df = pd.DataFrame(sizes)
    ax = sns.violinplot(data=df)
    # ax.set_xlabel("Edge Size", fontsize=12)
    ax.set_ylabel("Edge distribution", fontsize=12)
    fig = ax.get_figure()
    fig.savefig(head_path+"/edge_size_distribution_test_{0}.png".format(t_id))
    fig.clf()