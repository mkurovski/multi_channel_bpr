"""
Module with command line interface arguments for Argparser
"""
import argparse
import logging

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)

from multi_channel_bpr import __version__


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Multi Channel Bayesian Personalized Ranking")
    parser.add_argument(
        '--version',
        action='version',
        version='multi_channel_bpr {ver}'.format(ver=__version__))
    parser.add_argument(
        '-d',
        dest="d",
        help="latent feature dimension",
        type=int,
        metavar="INT",
        required=True)
    parser.add_argument(
        '-beta',
        nargs='+',
        dest="beta_list",
        help="share of unobserved within negative feedback",
        type=float,
        default=[1.],
        metavar="FLOAT")
    parser.add_argument(
        '-lr',
        dest="lr",
        help="learning rate",
        type=float,
        default=0.05,
        metavar="FLOAT")
    parser.add_argument(
        '-reg',
        nargs=3,
        dest="reg_param_list",
        help="regularization parameters for user, positive and negative item",
        type=float,
        default=[0.002]*3,
        metavar="FLOAT")
    parser.add_argument(
        '-k',
        dest="k",
        help="no. of items with highest predicted rating",
        type=int,
        metavar="INT",
        required=True)
    parser.add_argument(
        '-seed',
        dest="rd_seed",
        help="seed for random number generators",
        type=int,
        default=42,
        metavar="INT")
    parser.add_argument(
        '-folds',
        dest="n_folds",
        help="no. of folds for crossfold evaluation",
        type=int,
        default=4,
        metavar="INT")
    parser.add_argument(
        '-epochs',
        dest="n_epochs",
        help="no. of training epochs",
        type=int,
        default=10,
        metavar="INT")
    parser.add_argument(
        '-sampling',
        nargs='+',
        dest="neg_sampling_modes",
        help="list of negative item sampling modes",
        type=str,
        default=['uniform', 'non-uniform'],
        metavar="STR")
    parser.add_argument(
        '-data',
        dest="data_path",
        help="path to read MovieLens 1M input data from",
        type=str,
        default='../data/ml-1m',
        metavar="STR")
    parser.add_argument(
        '-results',
        dest="results_path",
        help="path to write results into",
        type=str,
        default='../results/',
        metavar="STR")
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)
