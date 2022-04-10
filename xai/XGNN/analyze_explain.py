import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

test_idx=[0,1,2,3,4,5,6,7,8,9]
head_path = "../../xai_results/original_XGNN_MUTAG_ckpt/max_node_5_max_step_20_max_iters_5"
for t_id in test_idx:
    Xs = []
    As = []
    probs = []
    targets = []
    sizes = []
    atom_cats = []
    for it in range(20):
        path = os.path.join(head_path, 'test_idx_{0}_iter_{1}.npz'.format(t_id, it))
        test = np.load(path)
        import pdb 
        pdb.set_trace()
        X = test['arr_0']
        A = test['arr_1']
        prob = test['arr_2']
        target = int(test['arr_3'])
        sizes.append(A.shape[0])
        Xs.append(X)
        As.append(A)
        probs.append(prob)
        targets.append(target)
        atom_cats.append(np.count_nonzero(X, axis=0))
    Xs = np.array(Xs)
    As = np.array(As)
    probs = np.array(probs)
    targets = np.array(targets)
    atom_cats = np.array(atom_cats)
    atoms = [0,1,2,3,4,5,6]
    df = pd.DataFrame(atom_cats,columns=atoms)
    ax = sns.violinplot(data=df)
    ax.set_xlabel("Atom type", fontsize=12)
    ax.set_ylabel("# atoms", fontsize=12)
    fig = ax.get_figure()
    fig.savefig("atom_type_{0}.png".format(t_id))
    fig.clf()
    df = pd.DataFrame(sizes)
    ax = sns.violinplot(data=df)
    ax.set_xlabel("Size", fontsize=12)
    ax.set_ylabel("Size variance", fontsize=12)
    fig = ax.get_figure()
    fig.savefig("size_{0}.png".format(t_id))
    fig.clf()

# size variance: 22.5275
# variance of atoms for each category: [17.55, 1.45, 2.99, 0.24, 0.32, 1.83, 0.29]