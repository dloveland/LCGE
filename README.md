# Notes

To train the GNN or explain the GNN run the main.py as: *python main.py -config_file config.yml*

The config.yml includes all configuration parameters from data read and storage to model hyperparameters.

This config file covers almost all hard-coded parameters for most XGNN files except policy_nn.py. Feel free to add.

I inserted the previous version of gcn.py for sanity check. Feel free to revert.

The requirements.txt file is changed based on my own configurations and python version, may need to merge.

# XGNN

# Usage


## XGNN Algorithm

The policy network of our XGNN is defined in "policy_nn.py". You can modify it as needed, depending on the GNNs/tasks at hand.

The explanation generation stage is defined in "gnn_explain.py". You can tune the hyper-parameters as needed. 

Simply call "main.py" to obtain explanations after proper settings and modifications. 

After training, the generated explanations should maximize the predictions of a certain class (or other targets). We found that there are multiple graph patterns that can maximize the predicted probability of a target class. 


## How to customize?

Our XGNN is a general framework, you can customize it for your own task. 

- Define your data/graph properties as needed. In this repository, we show how to explain a GCN classifier trained on the MUTAG dataset so each node is representing an atom. 

- Define your own graph rules in "gnn_explain.py". In our example, the check_validity function checks whether the generated graph is valid. 

- You can customize the roll_out function in "gnn_explain.py". For simple tasks on synthetic data, roll_out is not necessary. In addition, there are several ways to handle invalid generated graphs in the roll_out. In this example, we simply return a negative pre-defined reward. 

- The GNN layer, policy network architectures, and normalize_adj functions can be easily replaced by any suitable functions. 

- Our provided code is based on CPU so you can monitor the explanation generation step by step with IDEs, such as Spyder. 

- You can customize the target of explanations. Currently, our code explains the predictions of different classes. You may modify this to study what explanations activate other network targets, such as hidden neurons/ channels. 
