"""
Model initialization and training methods
"""

from collections import OrderedDict
import logging

from .evaluation import score_one_plus_random
from .sampling import (get_pos_channel, get_neg_channel,
                       get_pos_user_item, get_neg_item)
from .utils import *

__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class MultiChannelBPR:
    """
    Model Class for Multi-Channel Bayesian Personalized Ranking
    """
    def __init__(self, d, beta, rd_seed, channels, n_user, n_item, n_random=1000):
        """
        Args:
            d (int): no. of latent features for user and item representations
            beta (float): share of unobserved feedback within the overall
                negative feedback
            rd_seed (int): random number generator seed
            channels ([int]): rating values representing distinct feedback channels
            n_user (int): no. of unique users in the dataset
            n_item (int): no. of unique items in the dataset
            n_random (int): no. of items to sample for one-plus-random evaluation
        """
        self.d = d
        self.beta = beta
        self.rd_seed = rd_seed
        self.channels = channels
        self.n_user = n_user
        self.n_item = n_item
        self.n_random = n_random

    def set_train_data(self, train_ratings):
        """
        Attaches the training data to the model

        Args:
            train_ratings (:obj:`pd.DataFrame`): `M` training instances (rows)
                with three columns `[user, item, rating]`
        """
        self.train_inter_pos, self. train_inter_neg = \
            get_pos_neg_splits(train_ratings)
        self.pos_level_dist, self.neg_level_dist = \
            get_overall_level_distributions(self.train_inter_pos,
                                            self.train_inter_neg, self.beta)

        self.train_inter_pos_dict = get_pos_channel_item_dict(self.train_inter_pos)

        self.user_reps = get_user_reps(self.n_user, self.d, train_ratings,
                                       self.channels, self.beta)
        self.item_reps = get_item_reps(self.n_item, self.d)

    def fit(self, lr, reg_params, n_epochs, neg_item_sampling_mode, verbose=False):
        """
        Fits the model (user and item latent features)
        to the training data using the sampling
        approaches described in the paper by Loni et al.
        combined with BPR as the pairwise learning-to-rank approach
        proposed by Steffen Rendle

        Args:
            lr (float): learning rate for stochastic gradient descent
            reg_params (dict): three regularization parameters for user,
                positive item, and negative item latent feature updates
            n_epochs (int): no. of training epochs
            neg_item_sampling_mode (str): `uniform` or `non-uniform`
                mode to sample negative items
            verbose (bool): verbosity
        """
        n_examples = self.train_inter_pos.shape[0]
        # show result at every ~10% of the whole training data
        # TODO: Check verbosity differentiation in scikit-learn for training
        show_step = n_examples//10
        for epoch in range(n_epochs):
            for instance in range(n_examples):
                L = get_pos_channel(self.pos_level_dist)
                u, i = get_pos_user_item(L, self.train_inter_pos_dict)
                N = get_neg_channel(self.user_reps[u])
                j = get_neg_item(self.user_reps[u], N, self.n_item, u, i,
                                 self.train_inter_pos_dict,
                                 self.pos_level_dist,
                                 mode=neg_item_sampling_mode)
                user_embed, pos_item_embed, neg_item_embed = \
                    perform_gradient_descent(self.user_reps[u]['embed'],
                                             self.item_reps[i],
                                             self.item_reps[j],
                                             lr, reg_params)
                self.user_reps[u]['embed'] = user_embed
                self.item_reps[i] = pos_item_embed
                self.item_reps[j] = neg_item_embed
                if verbose:
                    if not instance % show_step:
                        self.print_learning_status(instance, n_examples)

    def predict(self, users, k):
        """
        Returns the `k` most-relevant items for every user in `users`

        Args:
            users ([int]): list of user ID numbers
            k (int): no. of most relevant items

        Returns:
            top_k_items (:obj:`np.array`): (len(users), k) array that holds
                the ID numbers for the `k` most relevant items
        """
        top_k_items = np.zeros((len(users), k))

        for idx, user in enumerate(users):
            user_embed = self.user_reps[user]
            pred_ratings = np.dot(self.item_reps, user_embed)
            user_items = np.argsort(pred_ratings)[::-1][:k]
            top_k_items[idx] = user_items

        return top_k_items

    def evaluate(self, test_ratings, k):
        """
        Offline evaluation of the model performance using precision,
        recall, and mean reciprocal rank computed for top-`k` positions
        and averaged across all users

        Args:
            test_ratings (:obj:`pd.DataFrame`): `M` testing instances (rows)
                with three columns `[user, item, rating]`
            `k` (int): no. of most relevant items

        Returns:
            result (tuple): mean average precision (MAP), mean average recall (MAR),
                and mean reciprocal rank (MRR) - all at `k` positions
        """
        result = score_one_plus_random(k, test_ratings,
                                       self.user_reps, self.item_reps)

        return result

    def print_learning_status(self, instance, n_examples):
        _logger.info("Step: %s/%s", str(instance), str(n_examples))


def get_pos_neg_splits(train_inter_df):
    """
    Calculates the rating mean for each user and splits the train
    ratings into positive (greater or equal as every user's
    mean rating) and negative ratings (smaller as mean ratings)

    Args:
        train_inter_df (:obj:`pd.DataFrame`): `M` training instances (rows)
            with three columns `[user, item, rating]`

    Returns:
        train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
            where `rating_{user}` >= `mean_rating_{user}
        train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
            where `rating_{user}` < `mean_rating_{user}
    """
    user_mean_ratings = \
        train_inter_df[['user', 'rating']].groupby('user').mean().reset_index()
    user_mean_ratings.rename(columns={'rating': 'mean_rating'},
                             inplace=True)

    train_inter_df = train_inter_df.merge(user_mean_ratings, on='user')
    train_inter_pos = train_inter_df[
        train_inter_df['rating'] >= train_inter_df['mean_rating']]
    train_inter_neg = train_inter_df[
        train_inter_df['rating'] < train_inter_df['mean_rating']]

    return train_inter_pos, train_inter_neg


def get_overall_level_distributions(train_inter_pos, train_inter_neg, beta):
    """
    Computes the frequency distributions for discrete ratings

    Args:
        train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
            where `rating_{user}` >= `mean_rating_{user}
        train_inter_neg (:obj:`pd.DataFrame`): training instances (rows)
            where `rating_{user}` < `mean_rating_{user}
        beta (float): share of unobserved feedback within the overall
            negative feedback

    Returns:
        pos_level_dist (dict): positive level sampling distribution
        neg_level_dist (dict): negative level sampling distribution
    """

    pos_counts = train_inter_pos['rating'].value_counts().sort_index(
            ascending=False)
    neg_counts = train_inter_neg['rating'].value_counts().sort_index(
            ascending=False)

    pos_level_dist = get_pos_level_dist(pos_counts.index.values,
                                        pos_counts.values)
    neg_level_dist = get_neg_level_dist(neg_counts.index.values,
                                        neg_counts.values, beta)

    return pos_level_dist, neg_level_dist


def get_pos_channel_item_dict(train_inter_pos):
    """
    Creates buckets for each possible rating in `train_inter_pos`
    and subsumes all observed (user, item) interactions with
    the respective rating within

    Args:
        train_inter_pos (:obj:`pd.DataFrame`): training instances (rows)
            where `rating_{user}` >= `mean_rating_{user}

    Returns:
        train_inter_pos_dict (dict): collection of all (user, item) interaction
            tuples for each positive feedback channel
    """

    pos_counts = train_inter_pos['rating'].value_counts().sort_index(
        ascending=False)
    train_inter_pos_dict = OrderedDict()

    for key in pos_counts.index.values:
        u_i_tuples = [tuple(x) for x in
                      train_inter_pos[train_inter_pos['rating'] == key][['user', 'item']].values]
        train_inter_pos_dict[key] = u_i_tuples

    return train_inter_pos_dict


def get_user_reps(m, d, train_inter, channels, beta):
    """
    Creates user representations that encompass user latent features
    and additional user-specific information
    User latent features are drawn from a standard normal distribution

    Args:
        m (int): no. of unique users in the dataset
        d (int): no. of latent features for user and item representations
        train_inter (:obj:`pd.DataFrame`): `M` training instances (rows)
            with three columns `[user, item, rating]`
        channels ([int]): rating values representing distinct feedback channels
        beta (float): share of unobserved feedback within the overall
            negative feedback

    Returns:
        user_reps (dict): representations for all `m` unique users
    """
    user_reps = {}
    train_inter = train_inter.sort_values('user')

    for user_id in range(m):
        user_reps[user_id] = {}
        user_reps[user_id]['embed'] = np.random.normal(size=(d,))
        user_item_ratings = train_inter[train_inter['user'] == user_id][['item', 'rating']]
        user_reps[user_id]['mean_rating'] = user_item_ratings['rating'].mean()
        user_reps[user_id]['items'] = list(user_item_ratings['item'])
        user_reps[user_id]['pos_channel_items'] = OrderedDict()
        user_reps[user_id]['neg_channel_items'] = OrderedDict()
        for channel in channels:
            if channel >= user_reps[user_id]['mean_rating']:
                user_reps[user_id]['pos_channel_items'][channel] = \
                    list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])
            else:
                user_reps[user_id]['neg_channel_items'][channel] = \
                    list(user_item_ratings[user_item_ratings['rating'] == channel]['item'])

        pos_channels = np.array(list(user_reps[user_id]['pos_channel_items'].keys()))
        neg_channels = np.array(list(user_reps[user_id]['neg_channel_items'].keys()))
        pos_channel_counts = [len(user_reps[user_id]['pos_channel_items'][key]) for key in pos_channels]
        neg_channel_counts = [len(user_reps[user_id]['neg_channel_items'][key]) for key in neg_channels]

        user_reps[user_id]['pos_channel_dist'] = \
            get_pos_level_dist(pos_channels, pos_channel_counts, 'non-uniform')

        if sum(neg_channel_counts) != 0:
            user_reps[user_id]['neg_channel_dist'] = \
                get_neg_level_dist(neg_channels, neg_channel_counts, 'non-uniform')

            # correct for beta
            for key in user_reps[user_id]['neg_channel_dist'].keys():
                user_reps[user_id]['neg_channel_dist'][key] = \
                    user_reps[user_id]['neg_channel_dist'][key] * (1 - beta)
            user_reps[user_id]['neg_channel_dist'][-1] = beta

        else:
            # if there is no negative feedback, only unobserved remains
            user_reps[user_id]['neg_channel_dist'] = {-1: 1.0}

    return user_reps


def get_item_reps(n, d):
    """
    Initializes item latent features from a standard normal distribution

    Args:
        n (int): no. of unique items in the dataset
        d (int): no. of latent features for user and item representations

    Returns:
        item_reps (:obj:`np.array`): (n, d) `d` latent features for all `n` items
    """
    item_reps = np.random.normal(size=(n, d))

    return item_reps


def perform_gradient_descent(user_embed, pos_item_embed, neg_item_embed, lr, reg_params):
    """
    Performs stochastic gradient descent using the BPR-Opt criterion
    and returns updated user as well as positive and negative item latent features

    Args:
        user_embed (:obj:`np.array`): (d,) `d` latent features for a single user
        pos_item_embed (:obj:`np.array`): (d,) `d` latent features for pos. item
        neg_item_embed (:obj:`np.array`): (d,) `d` latent features for neg. item
        lr (float): learning rate for stochastic gradient descent
        reg_params (dict): three regularization parameters for user,
            positive item, and negative item latent feature updates

    Returns:
        user_embed (np.array): (d,) latent user variables after update
        pos_item_embed (np.array): (d,) latent pos. item variables after update
        neg_item_embed (np.array): (d,) latent neg. item variables after update
    """
    d = user_embed.shape[0]
    lambda_u, lambda_i, lambda_j = reg_params['u'], reg_params['i'], reg_params['j']

    # 1. Step: Predict user rating for positive `i` and negative `j` item
    x_ui = np.dot(user_embed, pos_item_embed)
    x_uj = np.dot(user_embed, neg_item_embed)

    # 2. Step: Compute rating difference
    x_uij = x_ui - x_uj

    # 3. Step: Compute constant part of the difference derivative
    constant_term = np.exp(-x_uij) * sigmoid(x_uij)

    # 4. Step: Compute variable part of the difference derivative
    user_derivative = pos_item_embed - neg_item_embed
    pos_item_derivative = user_embed
    neg_item_derivative = -user_embed

    # 5. Step: Compute variable regularization part for the SGD update
    user_theta = rms(user_embed)
    pos_item_theta = rms(pos_item_embed)
    neg_item_theta = rms(neg_item_embed)

    # 6. Step: Combine all parts to get the full update for user and items
    user_update = lr * (constant_term * user_derivative + lambda_u * user_theta)
    pos_item_update = lr * (constant_term * pos_item_derivative + lambda_i * pos_item_theta)
    neg_item_update = lr * (constant_term * neg_item_derivative + lambda_j * neg_item_theta)

    # 7. Step: Perform the latent feature updates
    user_embed = user_embed + user_update
    pos_item_embed = pos_item_embed + pos_item_update
    neg_item_embed = neg_item_embed + neg_item_update

    return user_embed, pos_item_embed, neg_item_embed
