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
        pos_level_dist (dict): positive channel sampling distribution

    Returns:
        L (int): positive feedback channel
    """
    levels = list(pos_level_dist.keys())
    probabilities = list(pos_level_dist.values())
    L = np.random.choice(levels, p=probabilities)

    return L


def get_pos_user_item(L, train_inter_pos_dict):
    """
    Sample user u, positive feedback channel and item

    Args:
        L (int): positive feedback channel
        train_inter_pos_dict (dict): collection of all (user, item) interaction
                    tuples for each positive feedback channel

    Returns:
        (int, int, int): (u, i, L) user ID, positive item ID and feedback channel
    """
    pick_idx = np.random.randint(0, len(train_inter_pos_dict[L]))
    u, i = train_inter_pos_dict[L][pick_idx]

    return u, i


def get_neg_channel(user_rep):
    """
    Conditional negative level sampler
    Samples negative level based on the user-specific negative level
    sampling distribution (also influenced by beta)

    Args:
        user_rep (dict): user representation

    Returns:
        N (int): negative feedback channel
    """
    levels = list(user_rep['neg_channel_dist'].keys())
    probabilities = list(user_rep['neg_channel_dist'].values())
    N = np.random.choice(levels, p=probabilities)

    return N


def get_neg_item(user_rep, N, n, u, i, pos_level_dist, train_inter_pos_dict,
                 mode='uniform'):
    """
    Samples the negative item `j` to complete the update triplet `(u, i, j)

    If the sampled negative level `N` is actually an explicit negative channel,
    we sample uniformly from the items in the user's negative channel

    If the samples negative level `N` is actually the unobserved channel,
    we sample uniformly from all items the user did not interact with
    for mode == `uniform` and non-uniformly if the mode == `non-uniform`

    Args:
        user_rep (dict): user representation
        N (int): N (int): negative feedback channel
        n (int): no. of unique items in the dataset
        u (int): user ID
        i (int): positive item ID
        pos_level_dist (dict): positive channel sampling distribution
        train_inter_pos_dict (dict): collection of all (user, item) interaction
            tuples for each positive feedback channel
        mode (str): `uniform` or `non-uniform` mode to sample negative items

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
            pos_channel_interactions = train_inter_pos_dict[L]
            n_pos_interactions = len(pos_channel_interactions)
            pick_trials = 0  # ensure sampling despite
            u_other, i_other = u, i
            while u == u_other or i == i_other:
                pos_channel_interactions = train_inter_pos_dict[L]
                pick_idx = np.random.randint(n_pos_interactions)
                u_other, i_other = pos_channel_interactions[pick_idx]
                pick_trials += 1
                if pick_trials == 10:
                    # Ensures that while-loop terminates if sampled L does
                    # not provide properly different feedback
                    L = get_pos_channel(pos_level_dist)
                    pos_channel_interactions = train_inter_pos_dict[L]
                    n_pos_interactions = len(pos_channel_interactions)

            j = i_other

    return j
