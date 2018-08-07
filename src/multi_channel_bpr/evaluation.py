"""
Evaluation module
"""
import logging

import numpy as np

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def score_one_plus_random(k, test_inter, user_reps, item_reps, n_random=1000,
                          verbose=True):
    """
    Computes mean average precision, mean average recall and
    mean reciprocal rank based on the One-plus-random testing methodology
    outlined in
    "Performance of recommender algorithms on top-n recommendation tasks."
    by Cremonesi, Paolo, Yehuda Koren, and Roberto Turrin (2010)

    Args:
        k (int): no. of most relevant items
        test_inter (:obj:`pd.DataFrame`): `M` testing instances (rows)
            with three columns `[user, item, rating]`
        user_reps (dict): representations for all `m` unique users
        item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
        n_random (int): no. of unobserved items to sample randomly
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

    for idx in range(m):
        u = test_inter_red[idx, 0]
        i = test_inter_red[idx, 1]
        user_embed = user_reps[u]['embed']
        user_items = user_reps[u]['all_items']
        # 1. Randomly select `n_random` unobserved items
        random_uo_items = np.random.choice(np.setdiff1d(np.arange(n_item), user_items),
                                           replace=False, size=n_random)
        user_items = np.array(list(random_uo_items) + [i])
        user_item_reps = item_reps[user_items]
        # 2. Predict ratings for test item i and for unobserved items
        user_item_scores = np.dot(user_item_reps, user_embed)
        user_items = user_items[np.argsort(user_item_scores)[::-1][:k]]
        # 3. Get rank p of test item i within rating predictions
        i_idx = np.where(i == user_items)[0]
        if len(i_idx) != 0:
            # item i is among Top-N predictions
            n_hits += 1
            rr_agg += 1 / (i_idx[0] + 1)

        if verbose and (idx % (m//10) == 0):
            _logger.info("Evaluating %s/%s", str(idx), str(m))

    prec = n_hits / (m*k)
    rec = n_hits / m
    mrr = rr_agg / m

    return prec, rec, mrr
