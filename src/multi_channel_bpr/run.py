"""
Entry point for experiments on Bayesian Personalized Ranking
for Multi-Channel user feedback based on the paper
"Bayesian personalized ranking with multi-channel user feedback."
by Loni, Babak, et al.
in Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.
"""
from datetime import datetime
import logging
import os
import pickle
import sys

from sklearn.model_selection import KFold

from .cli import parse_args
from .model import MultiChannelBPR
from .utils import load_movielens, get_channels


__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Initialize MC-BPR Experiments...")
    train_verbose = True

    ratings, m, n = load_movielens(args.data_path)
    channels = get_channels(ratings)

    reg_params = {'u': args.reg_param_list[0],
                  'i': args.reg_param_list[0],
                  'j': args.reg_param_list[0]}

    kf = KFold(n_splits=args.n_folds, random_state=args.rd_seed, shuffle=True)
    split_count = 0
    res_dict = {}

    for train_index, test_index in kf.split(ratings):
        split_count += 1
        res_dict[split_count] = {}
        train_inter = ratings.iloc[train_index]
        test_inter = ratings.iloc[test_index]

        for neg_sampling_mode in args.neg_sampling_modes:
            res_dict[split_count][neg_sampling_mode] = {}

            for beta in args.beta_list:
                res_dict[split_count][neg_sampling_mode][beta] = {}
                _logger.info("Split/Sampling/Beta: {}/{} - {} - {}".format(
                        split_count, args.n_folds, neg_sampling_mode, beta
                ))

                model = MultiChannelBPR(d=args.d, beta=beta,
                                        rd_seed=args.rd_seed,
                                        channels=channels, n_user=m, n_item=n)
                model.set_data(train_inter, test_inter)
                _logger.info("Training ...")
                model.fit(lr=args.lr, reg_params=reg_params, n_epochs=args.n_epochs,
                          neg_item_sampling_mode=neg_sampling_mode,
                          verbose=train_verbose)
                _logger.info("Evaluating ...")
                prec, rec, mrr = model.evaluate(test_inter, args.k)

                res_dict[split_count][neg_sampling_mode][beta]['map'] = prec
                res_dict[split_count][neg_sampling_mode][beta]['mar'] = rec
                res_dict[split_count][neg_sampling_mode][beta]['mrr'] = mrr

        res_filename = datetime.strftime(datetime.now(), '%Y%m%d_bpr_multi_')
        res_filename = res_filename + str(split_count) + 'o' + str(args.n_folds) + '.pkl'
        res_filepath = os.path.join(args.results_path, res_filename)
        pickle.dump(res_dict[split_count], open(res_filepath, 'wb'))

    _logger.info("Experiments finished, results saved in %s", res_filepath)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
