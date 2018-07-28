"""
Model initialization and training methods
"""

from collections import OrderedDict

from .evaluation import score_one_plus_random
from .sampling import (get_pos_channel, get_neg_channel,
                        get_pos_user_item, get_neg_item)
from .utils import *

import logging
_logger = logging.getLogger(__name__)


class MultiChannelBPR:
    """

    """
    def __init__(self, d, beta, rd_seed, channels, n_user, n_item, n_random=1000):
        """
        Args:
            d (int): no. of embedding dimensions
            beta (float): share of unobserved within overall negative feedback
            rd_seed (int): random generator seed
            channels ([int]): rating values for distinct feedback channels
            n_user (int): no. of users in the dataset
            n_item (int): np. of items in the dataset
            n_random (int): items to sample for one-plus-random evaluation
        """
        self.d = d
        self.beta = beta
        self.rd_seed = rd_seed
        self.channels = channels
        self.n_user = n_user
        self.n_item = n_item
        self.n_random = n_random

    def set_train_data(self, train_ratings, beta):
        """
        Attaches the training data to the model

        Args:
            train_ratings (pd.DataFrame): training instances [user, item, rating]
            beta (float): share of unobserved feedback in the overall
                          negative feedback we are sampling from
        """
        self.train_inter_pos, self. train_inter_neg = \
            get_pos_neg_splits(train_ratings)
        self.pos_level_dist, self.neg_level_dist = \
            get_overall_level_distributions(self.train_inter_pos,
                                            self.train_inter_neg, beta)

        self.train_inter_pos_dict = get_pos_channel_item_dict(self.train_inter_pos)

        self.user_reps = get_user_reps(self.n_user, self.d, train_ratings,
                                       self.channels, beta)
        self.item_reps = get_item_reps(self.n_item, self.d)

    def fit(self, lr, reg_params, n_epochs, neg_item_sampling_mode, verbose=False):
        """

        Args:
            lr (float): learning rate
            reg_params (dict): regularization parameters for user, positive item,
                               and negative item latent feature updates
            n_epochs (int): no. of epochs to train on
            neg_item_sampling_mode (str):
            verbose (bool): verbosity
        """
        n_examples = self.train_inter_pos.shape[0]
        # show result at every ~1% of the whole training data
        show_step = n_examples//100
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

        Args:
            users ([int]): list of user IDs
            k (int): no. of best items to take

        Returns:
            top_k_items (np.array): len(users) x k dimensional array carrying
                                    the most relevant items
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

        Args:
            test_ratings (pd.DataFrame): test instances [user, item, rating]
            k (int): no. of best items to take

        Returns:
            result (tuple): (MAP, MAR, MRR)
        """
        result = score_one_plus_random(k, test_ratings,
                                       self.user_reps, self.item_reps)

        return result

    def print_learning_status(self, instance, n_examples):
        _logger.info("Step: %s/%s", str(instance), str(n_examples))


def get_pos_neg_splits(train_inter_df):

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

        user_reps[user_id]['pos_channel_dist'] = \
            get_pos_level_dist(pos_channels,
                               [len(user_reps[user_id]['pos_channel_items'][key]) for key in pos_channels],
                               mode='non-uniform')
        user_reps[user_id]['neg_channel_dist'] = \
            get_neg_level_dist(neg_channels,
                               [len(user_reps[user_id]['neg_channel_items'][key]) for key in neg_channels],
                               mode='non-uniform')

        # correct for beta
        for key in user_reps[user_id]['neg_channel_dist'].keys():
            user_reps[user_id]['neg_channel_dist'][key] = user_reps[user_id]['neg_channel_dist'][key] * (1 - beta)
        user_reps[user_id]['neg_channel_dist'][-1] = beta

    return user_reps


def get_item_reps(n, d):
    """

    """
    item_reps = np.random.normal(size=(n, d))

    return item_reps


def perform_gradient_descent(user_embed, pos_item_embed, neg_item_embed, lr, reg_param):
    """
    Performs stochastic gradient descent using the BPR-Opt criterion
    and returns updated user as well as positive and negative item latent variables

    Args:
        user_embed (np.array): (d,)
        pos_item_embed (np.array): (d,)
        neg_item_embed (np.array): (d,)
        lr (float): learning rate
        reg_param (dict): contains the regularization factors for user, pos item and neg item

    Returns:
        user_embed (np.array): (d,) latent user variables after update
        pos_item_embed (np.array): (d,) latent pos. item variables after update
        neg_item_embed (np.array): (d,) latent neg. item variables after update
    """
    d = user_embed.shape[0]
    lambda_u, lambda_i, lambda_j = reg_param['u'], reg_param['i'], reg_param['j']

    # 1. Step
    x_ui = np.dot(user_embed, pos_item_embed)
    x_uj = np.dot(user_embed, neg_item_embed)

    # 2. Step
    x_uij = x_ui - x_uj

    # 3. Step
    constant_term = np.exp(-x_uij) * sigmoid(x_uij)

    # 4. Step
    user_derivative = pos_item_embed - neg_item_embed
    pos_item_derivative = user_embed
    neg_item_derivative = -user_embed

    # 5. Step
    user_theta = rms(user_embed)
    pos_item_theta = rms(pos_item_embed)
    neg_item_theta = rms(neg_item_embed)

    # 6. Step
    # TODO: Put into separate method that also adds the terms to the embeddings
    user_update = lr * (constant_term * user_derivative + lambda_u * user_theta)
    pos_item_update = lr * (constant_term * pos_item_derivative + lambda_i * pos_item_theta)
    neg_item_update = lr * (constant_term * neg_item_derivative + lambda_j * neg_item_theta)

    # 7. Step
    user_embed = user_embed + user_update
    pos_item_embed = pos_item_embed + pos_item_update
    neg_item_embed = neg_item_embed + neg_item_update

    return user_embed, pos_item_embed, neg_item_embed