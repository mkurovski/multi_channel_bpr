# Recommender System Theory put into Practice
This repository contains Python implementations and combination of interesting scientific papers on recommender systems.

I will support my implementations with explanations on the theory using summaries and JuPyter Notebooks. Each section references a new paper which results I try to reproduce or apply to other datasets.

## Bayesian Personalized Ranking with Multi-channel User Feedback - Loni et al. (2016)

This paper builds upon the famous paper by Rendle on Bayesian Personalized Ranking for triplets `(u,i,j)` with a user prefering an item i over j. This pairwise learning-to-rank approach was extended to better capture the individual preferences regarding different feedback channels and preferences degreees associated with those channels.

### Usage
1. Clone the repository
2. Obtain the MovieLens 1M Dataset from [grouplens](https://grouplens.org/datasets/movielens/1m/) and unzip it
3. Change to the cloned folder `cd multi_channel_bpr`
4. Make sure you have `pandas`, `numpy` and `scikit-learn` installed in your environment
5. Install the package with `python setup.py install`
6. Call the package with the command `multi_channel_bpr` followed by the respective parameters:
	* `-d` (int): no. of latent features for user and item representations
	* `-beta` [(float)]: share of unobserved feedback within the overall negative feedback
	* `-lr` (float): learning rate for stochastic gradient descent
	* `-reg` [(float)]: regularization parameters for user, positive and negative item
	* `-k`(int): no. of most relevant items rating
	* `-rd_seed` (int): random number generator seed
	* `-folds` (int): no. of folds for crossfold evaluation
	* `-epochs` (int): no. of training epochs
	* `-sampling` [(str)]: list of negative item sampling modes, `uniform` and/or `non-uniform`
	* `-data` (str): path to read  input data from
	* `-results` (str): path to write results into

### Results

- [ ] run `multi_channel_bpr -v -d 50 -beta 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -k 10 -epochs 100 -v -sampling 'uniform' 'non-uniform' -seed 42` and report results

### Related Papers with implemented ideas
- [Loni, Babak, et al. **"Bayesian personalized ranking with multi-channel user feedback."** Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.](https://dl.acm.org/citation.cfm?id=2959163)
- [Rendle, Steffen, et al. **"BPR: Bayesian personalized ranking from implicit feedback."** Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.](https://arxiv.org/pdf/1205.2618.pdf)
- [Cremonesi, Paolo, Yehuda Koren, and Roberto Turrin. **"Performance of recommender algorithms on top-n recommendation tasks."** Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.](https://dl.acm.org/citation.cfm?id=1864721)

### Open Points
- [ ] Create JuPyter notebook to reproduce a single graph of the paper and use the different modules
- [ ] Add results of own experiments


## Backlog
- [Blondel, Mathieu, et al. "Higher-order factorization machines." Advances in Neural Information Processing Systems. 2016.](https://arxiv.org/pdf/1607.07195.pdf)
- [Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank recommendations with the k-order statistic loss." Proceedings of the 7th ACM conference on Recommender systems. ACM, 2013.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41534.pdf)
- [Eksombatchai, Chantat, et al. "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.](https://dl.acm.org/citation.cfm?id=3186183)

## Note
This project has been set up using PyScaffold 3.0.3. For details and usage
information on PyScaffold see [http://pyscaffold.org/](http://pyscaffold.org/).