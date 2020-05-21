import numpy as np
import torch
import time
# from pykeops.torch import LazyTensor
#from pykeops.numpy import LazyTensor



def kmeans_l1(x_bow, K):
  N, D = x_bow.shape
  x_bow_normalized = x_bow/torch.sum(x_bow, dim=1, keepdim=True).repeat(1,D)
  centers = x_bow_normalized[np.random.choice(N, K, replace=False), :]
  for ite in range(20):
    min_idx = torch.zeros(N)
    for n in range(N):
      min_idx[n] = torch.argmin(torch.sum((
          centers-x_bow[n:(n+1),:].repeat(K,1)).abs(), dim=1))
    for k in range(K):
      if torch.sum(min_idx==k)>0:
        centers[k,:] = torch.sum(
            x_bow_normalized[torch.nonzero(
            min_idx==k),:], dim=0)/torch.sum(min_idx==k).float().cuda()
      else:
        centers[k,:] = x_bow_normalized[np.random.choice(N, 1), :]
  centers_rank = np.argsort(torch.histc(min_idx, 
                                    bins=K, min=0, max=K-1).numpy())[::-1]
  return centers[centers_rank.copy(),:]

'''
def Kmeans_keops(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c

def KMeans_keops_numpy(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = np.copy(x[:K, :])  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster

        Ncl = np.bincount(cl).astype(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with np.bincount:
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
            Niter, end - start, Niter, (end - start) / Niter))

    return cl, c 
'''
def createTensors(bow, vocab):
  ''' Creates tensors and bag of words format for files to save time
      Args:
      bow - path to the numpy files
      vocab - path to vocab file
  '''
  t_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  x = np.load(bow)
  x_cnt = x[x.files[1]] # mutation count matrix
  x_idx = x[x.files[0]] # mutation index matrix
  N = len(x_idx)
  the_vocab = np.load(vocab)
  the_vocab = the_vocab[the_vocab.files[0]]

  x_bow = torch.zeros(N, the_vocab.shape[0], device=t_device)
  M = []
  for n in range(len(x_cnt)):
    x_cnt[n] = torch.from_numpy(x_cnt[n]).float().to(t_device)
    try:
      M.append(x_idx[n].size) # M is a list of document unique word counts.
    except:
      raise ValueError(x_idx[n], n) 
    x_bow[n, x_idx[n].tolist()] = x_cnt[n]

  torch.save(x_cnt, 'x_cnt_all.pt')
  torch.save(x_idx, 'x_idx_all.pt')
  torch.save(M, 'M_all.pt')
  torch.save(x_bow, 'x_bow_all.pt')
