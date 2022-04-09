import os
import configargparse


class ConfigurationParer():
    """This class defines customized configuration parser
    """

    def __init__(self,
                 config_file_parser_class=configargparse.YAMLConfigFileParser,
                 formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                 **kwargs):
        """This funtion decides config parser and formatter

        Keyword Arguments:
            config_file_parser_class {configargparse.ConfigFileParser} -- config file parser (default: {configargparse.YAMLConfigFileParser})
            formatter_class {configargparse.ArgumentDefaultsHelpFormatter} -- config formatter (default: {configargparse.ArgumentDefaultsHelpFormatter})
        """

        self.parser = configargparse.ArgumentParser(config_file_parser_class=config_file_parser_class,
                                                    formatter_class=formatter_class,
                                                    **kwargs)

    def add_save_cfgs(self):
        """This function adds saving path arguments: config file, model file...
        """

        # config file configurations
        group = self.parser.add_argument_group('Config-File')
        group.add('-config_file', '--config_file', required=False, is_config_file_arg=True, help='config file path')

        # model file configurations
        group = self.parser.add_argument_group('Model-File')
        group.add('-save_dir', '--save_dir', type=str, required=True, help='directory for saving checkpoints.')

    def add_data_cfgs(self):
        """This function adds dataset arguments: data file path...
        """

        # self.parser.add('-data_dir', '--data_dir', type=str, required=True, help='dataset directory.')
        self.parser.add('-data_file', '--data_file', type=str, required=True,
                        help='graph related data on nodes types and attributes.')
        self.parser.add('-dataset',
                        '--dataset',
                        type=str,
                        default='MUTAG',
                        help='Dataset name.')
        self.parser.add('-model_file', '--model_file', type=str, required=True, help='model data file.')
        self.parser.add('-train_gnn', '--train_gnn', action='store_true', help='train GNN')
        # self.parser.add('-adjacency_matrix_file',
        #                 '--adjacency_matrix_file',
        #                 type=str,
        #                 default='MUTAG_A.txt',
        #                 help='Adjacency file.')

        # self.parser.add('-graph_labels_file',
        #                 '--graph_labels_file',
        #                 type=str,
        #                 default='MUTAG_graph_labels.txt',
        #                 help='graph labels file.')

        # self.parser.add('-graph_indicator_file',
        #                 '--graph_indicator_file',
        #                 type=str,
        #                 default='MUTAG_graph_indicator.txt',
        #                 help='graph indicator file.')

        # self.parser.add('-node_labels_file',
        #                 '--node_labels_file',
        #                 type=str,
        #                 default='MUTAG_node_labels.txt',
        #                 help='node labels file.')

    def add_model_cfgs(self):
        """This function adds model (network) arguments: embedding, hidden unit...
        """

        # gnn training configurations
        group = self.parser.add_argument_group('GNN')
        group.add('-gnn_epochs',
                  '--gnn_epochs',
                  type=int,
                  default=2000,
                  help='number of epochs.')
        group.add('-mlp_hidden',
                  '--mlp_hidden',
                  type=int,
                  default=32,
                  help='mlp hidden size for the trained GNN.')
        
        group.add('-input_dim',
                  '--input_dim',
                  type=int,
                  # action='append',
                  # nargs='*',
                  help='input dim for training GNN.')
        group.add('-latent_dim',
                  '--latent_dim',
                  type=str,
                  # action='append',
                  # nargs='*',
                  help='latent dim list for trained GNN.')

        group = self.parser.add_argument_group('GNNExplainer')
        group.add('-max_nodes', '--max_nodes', type=int, default=2, help='maximum nodes of the produced graph.')
        group.add('-max_steps', '--max_steps', type=int, default=5, help='maximum steps to take when generating a graph.')
        group.add('-max_iters', '--max_iters', type=int, default=10,
                  help='maximum itertions to train the gnn explainer.')
        group.add('-num_classes', '--num_classes', type=int, default=2, help='number of classes.')
        group.add('-node_types', '--node_types', type=int, default=1, help='number of different types of nodes.')
        group.add('-target_class', '--target_class', type=int, default=1,
                  help='target class for which to generate explaination.')

    def add_optimizer_cfgs(self):
        """This function adds optimizer arguments
        """

        self.parser.add('-gnn_learning_rate',
                        '--gnn_learning_rate',
                        type=float,
                        default=0.005,
                        help='GNN learning rate.')
        self.parser.add('-gnn_momentum',
                        '--gnn_momentum',
                        type=float,
                        default=0.9,
                        help='GNN momentum.')

        self.parser.add('--gnn_weight_decay',
                        '-gnn_weight_decay',
                        type=float,
                        default=5e-4,
                        help="GNN weight decay.")

        self.parser.add('--learning_rate',
                        '-learning_rate',
                        type=float,
                        default=0.01,
                        help="learning rate for graph generation model.")
        self.parser.add('-roll_out_alpha',
                        '--roll_out_alpha',
                        type=int,
                        default=2,
                        help='roll_out_alpha for graph generation model.')
        self.parser.add('-roll_out_penalty',
                        '--roll_out_penalty',
                        type=float,
                        default=-0.1,
                        help='roll_out_penalty for graph generation model.')

        self.parser.add('-reward_stepwise',
                        '--reward_stepwise',
                        type=float,
                        default=0.1,
                        help='reward_stepwise for graph generation model.')

        self.parser.add('-start_from', '--start_from', type=str,
                        help = 'whether start from an existing graph or start from an empty graph')


        self.parser.add('-test_idx', '--test_idx', nargs='+', type=int,
                        help = 'if start from an existing graph, what is the test idx of the graph you want to load')        

    def parse_args(self):
        """This function parses arguments and initializes logger

        Returns:
            dict -- config arguments
        """

        cfg = self.parser.parse_args()

        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        # cfg.dataset_dir = os.path.join('datasets', cfg.data_dir)
        cfg.data_file = os.path.join('datasets', cfg.dataset, "data_file.json")

        cfg.model_checkpoints_dir = os.path.join(cfg.save_dir, 'model_checkpoints')
        if not os.path.exists(cfg.model_checkpoints_dir):
            os.makedirs(cfg.model_checkpoints_dir)

        return cfg

    def format_values(self):
        return self.parser.format_values()