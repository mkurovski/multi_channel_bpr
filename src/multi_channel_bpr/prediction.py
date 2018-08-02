"""
Module with prediction functions for model inference
"""
import logging

import numpy as np

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def get_top_k_recs(user_reps, item_reps, k):
    """
    For each user compute the `n` topmost-relevant items

    Args:
        user_reps (dict): representations for all `m` unique users
        item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
        k (int): no. of most relevant items

    Returns:
        item_recs ([[int]]): list of personalized recommendations for each user
            as lists of item IDs
    """
    n_user = len(user_reps)
    item_recs = []

    for u in range(n_user):
        user_embed = user_reps[u]['embed']
        user_item_scores = np.dot(item_reps, user_embed)
        item_recs.append(list(np.argsort(user_item_scores)[::-1][:k]))

    return item_recs
