import numpy as np
from sklearn import random_projection
import torch
from torch import nn
from torch.nn import functional as F
import faiss


# Kernel that interpolates between the mean for short distances and weighted inverse distance for large distances
def _mean_IDW_kernel(key, keys, opts):
    key = torch.unsqueeze(key, dim=1)
    keys = np.repeat(keys[np.newaxis,:,:], key.shape[0], axis=0)
    weights = 1 / (torch.square(torch.norm(key-keys, p=2, dim=-1)) + opts['delta'])
    return weights


def _norm_kNN_separator(weights, opts):
    top_values, top_idxs = torch.topk(weights, opts["num_neighbours"], dim=1)
    print(top_idxs.shape, "iiiiiiii")
    accessed = torch.zeros(weights.shape, dtype=int)
    accessed.scatter_(1, top_idxs, 1)
    weights[accessed==0] = 0
    weights /= weights.sum(dim=1, keepdim=True)
    return weights, accessed, top_idxs
    
def _softmax_separator(weights, opts):
    weights = F.softmax(weights * opts["beta"], dim=1)
    idxs = torch.ones_like(weights).nonzero()
    return weights, -weights, idxs

# k-nearest neighbours search
def _knn_search(queries, data, k, return_neighbours=False, res=None):
  num_queries, dim = queries.shape
  if res is None:
    dists, idxs = np.empty((num_queries, k), dtype=np.float32), np.empty((num_queries, k), dtype=np.int64)
    heaps = faiss.float_maxheap_array_t()
    heaps.k, heaps.nh = k, num_queries
    heaps.val, heaps.ids = faiss.swig_ptr(dists), faiss.swig_ptr(idxs)
    faiss.knn_L2sqr(faiss.swig_ptr(queries), faiss.swig_ptr(data), dim, num_queries, data.shape[0], heaps)
  else:
    dists, idxs = torch.empty(num_queries, k, dtype=torch.float32, device=queries.device), torch.empty(num_queries, k, dtype=torch.int64, device=queries.device)
    faiss.bruteForceKnn(res, faiss.METRIC_L2, faiss.cast_integer_to_float_ptr(data.storage().data_ptr() + data.storage_offset() * 4), data.is_contiguous(), data.shape[0], faiss.cast_integer_to_float_ptr(queries.storage().data_ptr() + queries.storage_offset() * 4), queries.is_contiguous(), num_queries, dim, k, faiss.cast_integer_to_float_ptr(dists.storage().data_ptr() + dists.storage_offset() * 4), faiss.cast_integer_to_long_ptr(idxs.storage().data_ptr() + idxs.storage_offset() * 8))
  if return_neighbours:
    neighbours = data[idxs.reshape(-1)].reshape(-1, k, dim)
    return dists, idxs, neighbours
  else:
    return dists, idxs

# # k-nearest neighbours search
# def _knn_search(queries, data, k, return_neighbours=False, res=None):
#   num_queries, dim = queries.shape
#   if res is None:
#     dists, idxs = np.empty((num_queries, k), dtype=np.float32), np.empty((num_queries, k), dtype=np.int64)
#     heaps = faiss.float_maxheap_array_t()
#     heaps.k, heaps.nh = k, num_queries
#     heaps.val, heaps.ids = faiss.swig_ptr(dists), faiss.swig_ptr(idxs)
#     faiss.knn_L2sqr(faiss.swig_ptr(queries), faiss.swig_ptr(data), dim, num_queries, data.shape[0], heaps)
#   else:
#     dists, idxs = torch.empty(num_queries, k, dtype=torch.float32, device=queries.device), torch.empty(num_queries, k, dtype=torch.int64, device=queries.device)
#     faiss.bruteForceKnn(res, faiss.METRIC_L2, faiss.cast_integer_to_float_ptr(data.storage().data_ptr() + data.storage_offset() * 4), data.is_contiguous(), data.shape[0], faiss.cast_integer_to_float_ptr(queries.storage().data_ptr() + queries.storage_offset() * 4), queries.is_contiguous(), num_queries, dim, k, faiss.cast_integer_to_float_ptr(dists.storage().data_ptr() + dists.storage_offset() * 4), faiss.cast_integer_to_long_ptr(idxs.storage().data_ptr() + idxs.storage_offset() * 8))
#   if return_neighbours:
#     neighbours = data[idxs.reshape(-1)].reshape(-1, k, dim)
#     return dists, idxs, neighbours
#   else:
#     return dists, idxs


# Dictionary-based memory (assumes key-value associations do not change)
class StaticDictionary(nn.Module):
  def __init__(self, args, hash_size):
    super().__init__()
    self.key_size = args.key_size
    self.num_neighbours = args.num_neighbours
    self.kernel = _mean_IDW_kernel
    self.kernel_opts = {'delta': args.kernel_delta}
    self.separator_opts = {'num_neighbours': args.num_neighbours, "beta": args.separation_beta}
    
    self.keys = 1e6 * torch.ones((args.dictionary_capacity, args.key_size), dtype=torch.float32, requires_grad=False).to(device=args.device)  # Add initial keys with very large magnitude values (infinity results in kNN returning -1 as indices)
    self.values = torch.zeros((args.dictionary_capacity, 1), dtype=torch.float32, requires_grad=False).to(device=args.device)
    self.hashes = 1e6 * np.ones((args.dictionary_capacity, hash_size), dtype=np.float32)  # Assumes hash of 1e6 will never appear TODO: Replace with an actual dictionary?
    self.last_access = torch.zeros(args.dictionary_capacity, dtype=torch.float32, requires_grad=False)


# Differentiable neural dictionary
class DND(StaticDictionary):
  def __init__(self, args, hash_size):
    super().__init__(args, hash_size)
    self.key_size = args.key_size
    self.alpha = args.dictionary_learning_rate
    # RMSprop components
    self.rmsprop_learning_rate, self.rmsprop_decay, self.rmsprop_epsilon = args.learning_rate, args.rmsprop_decay, args.rmsprop_epsilon
    self.rmsprop_keys_square_avg, self.rmsprop_values_square_avg = torch.zeros(args.dictionary_capacity, args.key_size), torch.zeros(args.dictionary_capacity, 1)

  # Lookup function
  def forward(self, key, learning=False):
    
    self.separator = _norm_kNN_separator
    
    # Use weighted average return over k nearest neighbours
    weights = self.kernel(key, self.keys, self.kernel_opts)  # Apply kernel function
    weights, accessed, idxs = self.separator(weights, self.separator_opts) # Apply separation function

    values = np.repeat(self.values[np.newaxis,:,:], key.shape[0], axis=0) # Retrieve values
    values = torch.tensor(values, requires_grad=True).to(device=key.device)
    output = torch.sum(weights * values[:,:,0], dim=1).unsqueeze(-1)
    print(idxs.shape, "idxs")
    values = values[accessed>0].reshape(key.shape[0], idxs.shape[1], 1)

    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += (1-accessed).sum(axis=0)
    if learning:
      neighbours = self.keys.unsqueeze(0).repeat(key.shape[0],1,1)[accessed>0].reshape(key.shape[0], idxs.shape[1], self.keys.shape[-1])
      return output, neighbours, values, idxs
    else:
      return output

  # Updates a batch of key-value pairs
  def update_batch(self, keys, values, hashes):
    # Test for matching states in batch
    sorted_idxs = np.argsort(values, axis=0)[::-1][:, 0]  # Sort values in descending order (max value first) TODO: Is this the way it should be done for NEC, or average?
    keys, values, hashes = keys[sorted_idxs], values[sorted_idxs], hashes[sorted_idxs]  # Rearrange keys, values and hashes in this order
    hashes, unique_indices = np.unique(hashes, axis=0, return_index=True)  # Retrieve unique hashes and indices of first occurences in array
    keys, values = keys[unique_indices], values[unique_indices]  # Extract corresponding keys and values

    # Perform hash check for exact matches
    dists, idxs = _knn_search(hashes, self.hashes, 1)  # TODO: Replace kNN search with real hash check
    dists, idxs = dists[:, 0], idxs[:, 0]
    match_idxs, non_match_idxs = np.nonzero(dists == 0)[0], np.nonzero(dists)[0]
    num_matches, num_non_matches = len(match_idxs), len(non_match_idxs)
    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += 1  # Increment last access for all items

    # Update exact match with Q-learning
    if num_matches > 0:
      idxs_match_idxs = idxs[match_idxs]
      self.keys[idxs_match_idxs] = keys[match_idxs] # Update keys (embedding may have changed)
      self.values[idxs_match_idxs] += self.alpha * (values[match_idxs] - self.values[idxs_match_idxs])
      # self.rmsprop_keys_square_avg[idxs_match_idxs], self.rmsprop_values_square_avg[idxs_match_idxs] = 0, 0  # TODO: Reset RMSprop stats here too?
      self.last_access[idxs_match_idxs] = 0
    
    # Otherwise add new states and n-step returns, replacing least recently updated entries
    if num_non_matches > 0:
      _, lru_idxs = torch.topk(self.last_access, num_non_matches)  # Find top-k LRU items
      self.keys[lru_idxs] = torch.from_numpy(keys[non_match_idxs]).to(device=self.keys.device)
      self.values[lru_idxs] =  torch.from_numpy(values[non_match_idxs]).to(device=self.values.device)
      self.hashes[lru_idxs] = hashes[non_match_idxs]
      self.last_access[lru_idxs] = 0
      self.rmsprop_keys_square_avg[lru_idxs], self.rmsprop_values_square_avg[lru_idxs] = 0, 0  # Reset RMSprop stats

  # Performs a sparse RMSprop update TODO: Add momentum option and gradient clipping?
  def gradient_update(self, keys, values, idxs):
    idxs, unique_idxs = np.unique(idxs.reshape(-1), return_index=True)  # Check for duplicates to remove
    keys, values = keys.reshape(-1, self.key_size)[unique_idxs], values.reshape(-1, 1)[unique_idxs]  # Remove duplicate keys and values
    if keys.grad is not None:
      grad = keys.grad.data
      square_avg = self.rmsprop_keys_square_avg[idxs]
      square_avg.mul_(self.rmsprop_decay).addcmul_(1 - self.rmsprop_decay, grad, grad)
      avg = square_avg.add(self.rmsprop_epsilon).sqrt_()
      keys.data.addcdiv_(-self.rmsprop_learning_rate, grad, avg)
      self.keys[idxs] = keys.detach().cpu().numpy()
      self.rmsprop_keys_square_avg[idxs] = square_avg
    if values.grad is not None:
      grad = values.grad.data
      square_avg = self.rmsprop_values_square_avg[idxs]
      square_avg.mul_(self.rmsprop_decay).addcmul_(1 - self.rmsprop_decay, grad, grad)
      avg = square_avg.add(self.rmsprop_epsilon).sqrt_()
      values.data.addcdiv_(-self.rmsprop_learning_rate, grad, avg)
      self.values[idxs] = values.detach().cpu().numpy()
      self.rmsprop_values_square_avg[idxs] = square_avg


class NEC(nn.Module):
  def __init__(self, args, observation_shape, action_space, hash_size):
    super().__init__()
    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_keys = nn.Linear(3136, args.key_size)
    self.memories = [DND(args, hash_size) for _ in range(action_space)]

  def forward(self, observation, learning=False):
    hidden = F.relu(self.conv1(observation))
    hidden = F.relu(self.conv2(hidden))
    hidden = F.relu(self.conv3(hidden))
    keys = self.fc_keys(hidden.view(-1, 3136))
    memory_output = [memory(keys, learning) for memory in self.memories]
    if learning:
      memory_output, neighbours, values, idxs = zip(*memory_output)
      return torch.cat(memory_output, dim=1), neighbours, values, idxs, keys  # Return Q-values, neighbours, values and keys
    else:
      return torch.cat(memory_output, dim=1), keys  # Return Q-values and keys
