# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:28:06 2020

@author: orteg
"""
import math
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import beta
from sklearn.preprocessing import quantile_transform
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import cosine_similarity


def make_noisy_negatives(y, 
                         X = None, 
                         flip_ratio = None, 
                         label_noise = 'uniform',
                         n_neighbors = 'auto',
                         true_prop_pos = None, 
                         pollution_ratio = None,
                         random_state = None):
  
  """ Flipping Negatives into Positives
  
    Generating noisy negatives by flipping. It can generate noisy negatives 
    using a flip ratio (percentage of flipped positives) or a pollution ratio 
    (percentage of hidden positives in negatives)
    
    Parameters
    ----------
    y                 : array-like, compulsory
    X                 : array-like, sparse matrix of shape (n_samples, n_features)
    flip_ratio        : float, optional (default = None)
        Percentage of positives into noisy negatives.
    true_prop_pos     : float, optional (default = None)
        Percentage of true positives in the dataset.
    pollution_ratio   : float, optional (default = None)
        Percentage of hidden positives (noisy negs) among negatives.
    random_seed       : int, optional (default = 123)
      
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    """
  if X is not None:  
    X = check_array(X, accept_sparse='csr', dtype=np.float64)
  
  if X is None and label_noise == 'knn':
    raise ValueError("Label noise by Nearest Neighbors requires features matrix ``X`` ")
  
  random_state = check_random_state(random_state)
  
  y_ = y.copy()
  y_ = check_array(y_, ensure_2d=False, dtype=None)
  check_consistent_length(X, y_)
  n_pos = np.sum(y_ == 1)
  
  true_prop_pos = np.mean(y_)
  
  if n_neighbors == 'auto':
    n_neighbors = math.floor(math.sqrt(len(y_)))
    
  if flip_ratio is None and pollution_ratio is None:
    raise ValueError("Either flip rate (`flip_ratio`) or pollution ratio (`pollution_ratio`) must be known")
 
  if flip_ratio is not None and true_prop_pos is not None:
    flip_ratio_ = flip_ratio
       
  else:
    raise ValueError("If flip ratio is not None, true proportion of positives must be known")

  ## Types of Label Noise
  
  # Uniform Label Noise
  if label_noise == 'uniform':
    p_ = None

  # KNN Label Noise
  elif label_noise == 'knn':
    

    dist_matrix = pairwise_distances(X, n_jobs = -1)
    ix_pos = np.where(y == 1)[0]
    ix_neg = np.where(y == 0)[0]

    pos_dist_mt = dist_matrix[ix_pos][:,ix_neg]
    
    pos_dist_mt.sort(axis = 1)
    k_mean_dist = np.mean(pos_dist_mt[:,:n_neighbors], axis = 1)
    
    sample_weight = np.array([k_mean_dist[i]/np.sum(k_mean_dist) for i,_ in enumerate(k_mean_dist)])
    p_ = sample_weight.reshape(-1,)
    
    
  else: 
    raise ValueError('labeling is not valid; Use knn or uniform instead')
    
  ## Size of Flipped Negatives
  if label_noise == 'knn':
    size_ = min(int(np.ceil(flip_ratio_*n_pos)), len(np.nonzero(p_)[0]))  
    
  else: 
    size_ = int(np.ceil(flip_ratio_*n_pos))
  
  ## Index to be Flipped Positives
  indices = random_state.choice(range(n_pos), 
                                size_, 
                                replace = False, 
                                p = p_)
    
  print('Number Samples First Stage ', len(indices))

  if len(indices) < int(np.ceil(flip_ratio_*n_pos)):

      print('Sampling Second Stage - Random Sampling on Complement Index Set')

      indices_comp = np.setdiff1d(range(n_pos), indices)
      size_extra_ = int(np.ceil(flip_ratio_*n_pos)) - len(indices)
      indices_extra = random_state.choice(indices_comp,
                                          size_extra_,
                                          replace=False)
      
      print('Number Samples Second Stage ', len(indices_extra))

      indices = np.hstack((indices, indices_extra))
    
  
  pos_y = y_[y_ == 1].copy()
  pos_y[indices] = 0
  
  y_[y_ == 1] = pos_y

  mislabeled_rate = 1 - np.mean(pos_y)
  pol_ratio = (mislabeled_rate*true_prop_pos)/(1 - true_prop_pos*(1 - mislabeled_rate))

  print("Flipping ratio (pos. to neg.): {:.3f}".format(mislabeled_rate))
  print("Pollution ratio (noisy neg. among observed neg.): {:.3f}".format(pol_ratio))
  
  return y_
