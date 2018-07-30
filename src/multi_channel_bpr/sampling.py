"""
Module with channel and item sampling functions
"""
import logging

import numpy as np

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def get_pos_channel(pos_level_dist):
    """
    Samples a positive feedback channel

    Args:
        pos_level_dist (dict): sampling distribution to use

    Returns:
        L (str): positive feedback channel represented by rating level
    """
    levels = list(pos_level_dist.keys())
    probabilities = list(pos_level_dist.values())
    L = np.random.choice(levels, p=probabilities)

    return L


def get_pos_user_item(L, train_inter_pos_dict):
    """
    Sample user u, positive feedback channel and item

    Args:
        L (int): positive feedback channel represented by rating level
        train_inter_pos_dict (dict):

    Returns:
        (int, int, str): (u, i, L) user, positive item ID and feedback channel
    """
    pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
    u, i = train_inter_pos_dict[L][pick_idx]

    return u, i


def get_neg_channel(user_rep):
    """
    Sample negative channel from user-specific channel distribution

    Args:
        user_rep (dict): user representation

    Returns:
        L (str): negative feedback channel represented by rating level
    """
    levels = list(user_rep['neg_channel_dist'].keys())
    probabilities = list(user_rep['neg_channel_dist'].values())
    N = np.random.choice(levels, p=probabilities)

    return N


def get_neg_item(user_rep, N, n, u, i, pos_level_dist, train_inter_pos_dict,
                 mode='uniform'):
    """
    Sample negative item for a given user using the
    provided unobserved (-1) or negative channel, positive item and
    user-specific interaction information

    Args:
        user_rep (dict): user representation
        N (str): negative feedback channel represented by rating level
        n (int): no. of unique items in the dataset
        u (int): user ID
        i (int): positive item ID
        pos_level_dist (dict):
        train_inter_pos_dict (dict):
        mode (str):

    Returns:
        j (int): sampled negative item ID
    """
    if N != -1:
        # sample uniformly from negative channel
        neg_items = list(user_rep['neg_channel_items'][N])
        j = np.random.choice(neg_items)

    else:
        if mode == 'uniform':
            # sample item uniformly from unobserved channel
            j = np.random.choice(np.setdiff1d(np.arange(n), user_rep['items']))
        elif mode == 'non-uniform':
            # sample item non-uniformly from unobserved channel
            L = get_pos_channel(pos_level_dist)
            u_else, i_else = u, i
            while u == u_else or i == i_else:
                pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
                u_else, i_else = train_inter_pos_dict[L][pick_idx]
            j = i_else

    return j
