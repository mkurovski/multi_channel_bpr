"""
Evaluation module
"""

import numpy as np
import pandas as pd

import logging
_logger = logging.getLogger(__name__)


def score_one_plus_random(k, user_reps, item_reps, test_inter, n_random=1000,
                          verbose=True):
    """
    Computes mean average precision, mean average recall and
    mean reciprocal rank based on the One-plus-random testing methodology
    outlined in
    "Performance of recommender algorithms on top-n recommendation tasks."
    by Cremonesi, Paolo, Yehuda Koren, and Roberto Turrin (2010)

    Args:
        k (int): no. of ranked prediction positions to consider
        user_reps (dict): user representations
        item_reps (np.array): item latent features
        test_inter (pd.DataFrame): DataFrame with [user, item, rating] entries
        n_random (int): no. of random unobserved items to sample
        verbose (bool): verbosity

    Returns:
        prec (float): mean average precision @ k
        rec (float): mean average recall @ k
        mrr (float): mean reciprocal rank @ k
    """
    test_inter_red = test_inter[test_inter['rating'] == 5]
    test_inter_red = test_inter_red[['user', 'item']].values

    n_hits = 0
    rr_agg = 0
    m = test_inter_red.shape[0]
    n_item = item_reps.shape[0]

    for i in range(m):
        u = test_inter_red[i, 0]
        i = test_inter_red[i, 1]
        user_embed = user_reps[u]['embed']
        user_items = user_reps[u]['items']
        # 1. Randomly select 1000 additional items unrated by the user u
        # Assume that most of them will not be of interest to user u
        random_uo_items = np.random.choice(np.setdiff1d(np.arange(n_item), user_items),
                                           replace=False, size=n_random)
        user_items = np.array(list(random_uo_items) + [i])
        user_item_reps = item_reps[user_items]
        # 2. predict the ratings for the test item i and for the additional 1000 items
        user_item_scores = np.dot(user_item_reps, user_embed)
        user_items = user_items[np.argsort(user_item_scores)[::-1][:k]]
        # 3. Get rank p of the test item i within this list
        idx = np.where(i == user_items)[0]
        if len(idx) != 0:
            # item i is among Top-N predictions
            n_hits += 1
            rr_agg += 1 / (idx[0] + 1)

        if not (i % (m // 10)) & verbose:
            _logger.info("Evaluating %i/%i", i, m)

    prec = n_hits / (m*k)
    rec = n_hits / m
    mrr = rr_agg / m

    return prec, rec, mrr
