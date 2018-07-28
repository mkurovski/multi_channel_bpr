"""
Helper functions
"""

import os

import numpy as np
import pandas as pd

import logging
_logger = logging.getLogger(__name__)


def get_pos_level_dist(weights, level_counts, mode='non-uniform'):
    """
    Returns the sampling distribution for the
    different feedback channels L
    weights refer to levels

    Args:
        weights (np.array):
        level_counts (np.array):
        mode (str):

    Returns:
        dist (dict): (weight, probability)-pairs
    """
    if mode == 'non-uniform':
        nominators = weights * level_counts
        denominator = sum(nominators)
        dist = nominators / denominator
    else:
        n_levels = len(weights)
        dist = np.ones(n_levels) / n_levels

    dist = dict(zip(list(weights), dist))

    return dist


def get_neg_level_dist(weights, level_counts, mode='non-uniform'):
    """

    """
    if mode == 'non-uniform':
        nominators = [weight * count for weight, count in zip(weights, level_counts)]
        denominator = sum(nominators)
        if denominator != 0:
            dist = list(nom / denominator for nom in nominators)
        else:
            dist = [0] * len(nominators)
    else:
        n_levels = len(weights)
        dist = [1 / n_levels] * n_levels

    dist = dict(zip(list(weights), dist))

    return dist


def rms(x):
    """
    Calculates Root Mean Square for array x
    """
    s = np.square(x)
    ms = np.mean(s)
    result = np.sqrt(ms)

    return result


def sigmoid(x):
    """
    Calculates Sigmoid of x
    """
    return 1/(1+np.exp(-x))


def load_movielens(path):
    """
    loads the movielens 1M dataset, solely interactions

    Args:
        path (str): path pointing to folder with interaction data `ratings.dat

    Returns:
        ratings (pd.DataFrame): interactions (user, item, rating)
        m (int): no. of users
        n (int): no. of items
    """

    ratings = pd.read_csv(os.path.join(path, 'ratings.dat'), sep='::', header=0,
                          names=['user', 'item', 'rating', 'timestamp'])
    ratings.drop('timestamp', axis=1, inplace=True)

    m = ratings['user'].unique().shape[0]
    n = ratings['item'].unique().shape[0]

    user_rehasher = dict(zip(ratings['user'].unique(), np.arange(m)))
    item_rehasher = dict(zip(ratings['item'].unique(), np.arange(n)))
    ratings['user'] = ratings['user'].map(user_rehasher).astype(int)
    ratings['item'] = ratings['item'].map(item_rehasher)

    return ratings, m, n


def get_channels(inter_df):
    """
    Return existing feedback channels ordered
    by descending preference level

    Args:
        inter_df (pd.DataFrame):
    Returns:
        channels (float):
    """
    channels = list(inter_df['rating'].unique())
    channels.sort()

    return channels[::-1]
