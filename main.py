from utils.argparse import ConfigurationParer
from xai.XGNN.gnn_explain import gnn_explain
from xai.continuous_XGNN.continuous_relax import continuous_gnn_explain
from training.train_gnn import train
import logging
import json

logger = logging.getLogger(__name__)


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    cfg = parser.parse_args()
    logger.info(parser.format_values())
    # train GNN
    if cfg.train_gnn:
        train(cfg)

    # explain GNN through XGNN
    #gnn_explainer = gnn_explain(cfg)
    #gnn_explainer.train(cfg.model_checkpoints_dir, cfg.model_file)

    # Continuous relax to explain
    gnn_explainer = continuous_gnn_explain(cfg)
    gnn_explainer.train(cfg.model_checkpoints_dir, cfg.model_file)
   

if __name__ == '__main__':
    main()


