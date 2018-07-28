"""
Module with prediction functions for model inference
"""

import numpy as np

import logging
_logger = logging.getLogger(__name__)


def get_top_k_recs(user_reps, item_reps, k):
    """
    For each user compute the top-n item recommendations

    Args:
        user_reps (dict): user representations
        item_reps (np.array): item latent features
        k (int): no. of best items to take

    Returns:
        item_recs ([[int]]): list of personalized recommendations for each user
                             resembling lists of item IDs
    """
    n_user = len(user_reps)
    item_recs = []

    for u in range(n_user):
        user_embed = user_reps[u]['embed']
        user_item_scores = np.dot(item_reps, user_embed)
        item_recs.append(list(np.argsort(user_item_scores)[::-1][:k]))

    return item_recs
