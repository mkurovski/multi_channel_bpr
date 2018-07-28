"""
Entry point for experiments on Bayesian Personalized Ranking
for Multi-Channel user feedback based on the paper
"Bayesian personalized ranking with multi-channel user feedback."
by Loni, Babak, et al.
in Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
"""

from datetime import datetime
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import KFold

from .model import MultiChannelBPR
from .utils import load_movielens, get_channels

import logging
_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


setup_logging(logging.INFO)


def main():
    """
    Runs the experiment
    """
    # TODO: Explicate the parameters
    movielens_path = '../data/ml-1m'
    results_path = '../results/'
    n_folds = 4
    rd_seed = 42
    neg_sampling_modes = ['uniform', 'non-uniform']

    ratings, m, n = load_movielens(movielens_path)
    channels = get_channels(ratings)

    kf = KFold(n_splits=n_folds, random_state=rd_seed, shuffle=True)

    beta_list = np.linspace(0,1,11)

    d = 50
    lr = 0.05
    reg_params = {'u': 0.002, 'i': 0.002, 'j': 0.002}
    n_epochs = 1
    k = 10

    split_count = 0

    train_verbose = True

    res_dict = {}

    for train_index, test_index in kf.split(ratings):
        split_count += 1
        res_dict[split_count] = {}
        train_inter = ratings.iloc[train_index]
        test_inter = ratings.iloc[test_index]

        for neg_sampling_mode in neg_sampling_modes:
            res_dict[split_count][neg_sampling_mode] = {}

            for beta in beta_list:
                res_dict[split_count][neg_sampling_mode][beta] = {}
                _logger.info("Split/Sampling/Beta: {}/{} - {} - {}".format(
                        split_count, n_folds, neg_sampling_mode, beta
                ))

                model = MultiChannelBPR(d=d, beta=beta, rd_seed=rd_seed,
                                        channels=channels, n_user=m, n_item=n)
                model.set_train_data(train_inter, beta=beta)
                _logger.info("Training ...")
                model.fit(lr=lr, reg_params=reg_params, n_epochs=n_epochs,
                          neg_item_sampling_mode=neg_sampling_mode,
                          verbose=train_verbose)
                _logger.info("Evaluating ...")
                prec, rec, mrr = model.evaluate(test_inter, k)

                res_dict[split_count][neg_sampling_mode][beta]['map'] = prec
                res_dict[split_count][neg_sampling_mode][beta]['mar'] = rec
                res_dict[split_count][neg_sampling_mode][beta]['mrr'] = mrr

        # TODO: Persist results
        res_filename = datetime.strftime(datetime.now(), '%Y%m%d_bpr_multi_')
        res_filename = res_filename + str(split_count) + 'o' + str(n_folds) + '.pkl'
        res_filepath = os.path.join(results_path, res_filename)
        pickle.dump(res_dict[split_count], open(res_filepath, 'wb'))


if __name__ == '__main__':
    main()
