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
    Non-uniformly samples a channel from the provided distribution
    """
    levels = list(pos_level_dist.keys())
    probabilities = list(pos_level_dist.values())
    L = np.random.choice(levels, p=probabilities)

    return L


def get_pos_user_item(L, train_inter_pos_dict):
    """
    Sample user u, positive feedback channel and item

    Args:
        L (int):
        train_inter_pos_dict (dict):

    Returns:
        (int, int, str): (u, i, L) tuple
    """
    # uniformly pick (u, i) from chosen channel L
    pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
    u, i = train_inter_pos_dict[L][pick_idx]

    return u, i


def get_neg_channel(user_rep):
    """
    Sample negative channel from user-specific channel distribution
    """
    levels = list(user_rep['neg_channel_dist'].keys())
    probabilities = list(user_rep['neg_channel_dist'].values())
    N = np.random.choice(levels, p=probabilities)

    return N


def get_neg_item(user_rep, N, n, u, i, pos_level_dist, train_inter_pos_dict,
                 mode='uniform'):
    """
    Samples negative item from user rep
    provided unobserved (-1) or negative channel

    Args:
        user_rep:
        N:
        n:
        u:
        i:
        pos_level_dist:
        train_inter_pos_dict:
        mode:

    Returns:

    """
    if N != -1:
        # sample uniformly from negative channel
        neg_items = list(user_rep['neg_channel_items'][N])
        j = np.random.choice(neg_items)

    else:
        if mode == 'uniform':
            # sample uniformly from unobserved channel
            j = np.random.choice(np.setdiff1d(np.arange(n), user_rep['items']))
        elif mode == 'non-uniform':
            # sample non-uniformly from unobserved channel
            # taking into account popular items with high relevance for other user u'
            L = get_pos_channel(pos_level_dist)
            u_else, i_else = u, i
            while u == u_else or i == i_else:
                pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
                u_else, i_else = train_inter_pos_dict[L][pick_idx]
            j = i_else

    return j
