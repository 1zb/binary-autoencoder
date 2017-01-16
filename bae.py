from __future__ import print_function, nested_scopes, unicode_literals, division

import numpy as np
# import matplotlib.pyplot as plt
import sys
# from scipy.sparse import dok_matrix,coo_matrix,csr_matrix
# from random import choice, shuffle
import timeit
import itertools

sys.path.append('../liblinear/python')
import liblinearutil

if sys.version_info < (3, 5):
    import cPickle
else:
    import pickle as cPickle

def read_cifar(path='../cifar-10-batches-py/'):
  """ load cifar images
  """
  files = ['data_batch_1',
           'data_batch_2',
           'data_batch_3',
           'data_batch_4',
           'data_batch_5',
           'test_batch',]

  images = []
  labels = []
  for file in files:
    with open(path + file, 'rb') as fo:
      if sys.version_info < (3, 5):
        dict=cPickle.load(fo)
      else:
        dict=cPickle.load(fo, encoding='bytes')
      images.append(dict[b'data'].reshape(-1, 3, 32, 32).transpose(0,2,3,1))
      labels.append(np.asarray(dict[b'labels']).reshape(-1, 1))


  images = np.vstack(images)
  labels = np.vstack(labels).reshape(-1)
  return (images, labels)

def load_gist():
  """ load gist features
  """
  with open('../triplet_hashing-master/features', 'rb') as fo:
    features=cPickle.load(fo)
  return features


def h_step(features, codes, verbose=True):
  N, D = features.shape
  models = []
  for (y, i) in zip(codes.T, range(codes.shape[1])):
    t_start = timeit.default_timer()
    models.append(liblinearutil.train(y.tolist(), features.tolist(), str('-s 0 -c 4 -q')))
    t_end = timeit.default_timer()
    if verbose:
      print('[H] {:3d}th bit, {:.4f} seconds elapsed'.format(i, t_end-t_start))
  return models

def f_step(features, models, verbose=True):
  # X = features
  Z = []
  for (m, i) in zip(models, range(len(models))):
    t_start = timeit.default_timer()
    p_label, p_acc, p_val = liblinearutil.predict([0]*features.shape[0], features.tolist(), m , str('-q'))
    Z.append(p_label)
    t_end = timeit.default_timer()
    if verbose:
      print('[F] {:3d}th bit, {:.4f} seconds elapsed'.format(i, t_end-t_start))

  Z = np.vstack(Z).transpose()

  # np.linalg.pinv(Z).dot(X).shape
  return (np.linalg.pinv(Z).dot(features), Z)

def generate_enums(L):
  enums = []
  for i in range(L+1):
    for subset in itertools.combinations(range(L),i):
      subset=np.array(subset,dtype=int)
      enum=np.zeros(L,dtype=int)
      enum[subset]=1
      enums.append(enum)
  enums = np.vstack(enums)
  return enums

def z_step(features, models, A, old_Z, mu):
  assert old_Z.shape[1] == len(models)

  t_start = timeit.default_timer()
  L = old_Z.shape[1]
  enums = generate_enums(L)

  loss = []
  for enum in enums:
    loss.append(np.linalg.norm(features-enum.dot(A), axis=1) ** 2 + mu * np.linalg.norm(old_Z-enum, axis=1) ** 2)

  loss = np.vstack(loss)
  min_idx = np.argmin(loss, axis=0)
  # print(sum(np.min(loss, axis=0)))

  t_end = timeit.default_timer()
  print('[Z] {:.4f} seconds elapsed'.format(t_end-t_start))

  return (enums[min_idx], sum(np.min(loss, axis=0)))

def test_recon(features, models, A):
  _, Z = f_step(features, models, verbose=False)
  recon_error = sum(np.linalg.norm(features-Z.dot(A), axis=1) ** 2) / features.shape[0]
  return recon_error

def hash(features, num_train_samples=58000, L=8):
  bits = []
  for i in range(L):
    start = timeit.default_timer()
    m = liblinearutil.load_model('models/tr{0:05d}-L{1:02d}-b{2:02d}.model'.format(num_train_samples, L, i))
    p_label, p_acc, p_val = liblinearutil.predict([0]*features.shape[0], features.tolist(), m , str('-q'))
    bits.append(p_label)
    end = timeit.default_timer()
    print('[HASH] {0:3d}th bit hashed. {1:.4f} seconds elapsed'.format(i, end-start))

  start = timeit.default_timer()
  bits = np.vstack(bits).transpose().astype(np.int)
  bits[np.nonzero(bits==0)] = -1

  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'wb') as fo:
    cPickle.dump(bits, fo)
  end = timeit.default_timer()
  print('[HASH] Hash codes saved. {0:.4f} seconds elapsed'.format(end-start))
  return

def calc_mean_ap(base_set_labels, num_test, num_train_samples=58000, L=8):
  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'rb') as fo:
    codes = cPickle.load(fo)

  assert codes.shape[0]==base_set_labels.shape[0]

  test_labels = base_set_labels[-num_test:]

  distances = -codes[-num_test:].dot(codes.transpose())

  min_idx = np.argsort(distances)
  mean_ap = 0.0
  for i in range(num_test):
    counter = 0
    ap = 0.0
    for j in range(500):
      if base_set_labels[min_idx[i,j]]==test_labels[i]:
        counter = counter + 1
        ap = ap + counter / (j + 1.0)
    if counter == 0:
      counter = 1
    ap = ap / counter
    mean_ap = mean_ap + ap
  mean_ap = mean_ap / num_test

  return mean_ap

def calc_precision_at_k(base_set_labels, num_test, num_train_samples=58000, L=8, K=50):
  with open('hash/tr{0:05d}-L{1:02d}'.format(num_train_samples, L), 'rb') as fo:
    codes = cPickle.load(fo)

  assert codes.shape[0]==base_set_labels.shape[0]

  test_labels = base_set_labels[-num_test:]

  distances = -codes[-num_test:].dot(codes.transpose())

  min_idx = np.argsort(distances)

  p = 0.0
  for i in range(num_test):
    counter = 0
    for j in range(K):
      if base_set_labels[min_idx[i,j]]==test_labels[i]:
        counter = counter + 1
    p = p + counter / (K * 1.0)
  p = p / num_test
  return p

if __name__ == '__main__':
  (color_images, labels) = read_cifar('../data/cifar-10-batches-py/')
  # mu = np.array([0,0,0,0,.0005,.0005,.0005,.0005,.001,.001,.001,.002,.002,.005,.005,.01,.01,.05,.05,.2])
  mu = np.array([0.0005, 0.001, 0.01, 0.02, 0.05, 0.1])

  features = load_gist()
  num_train = 1000
  num_test = 2000
  train_features = features[:num_train]
  test_features = features[-num_test:]

  L = 12

  codes = np.random.randint(2, size=(train_features.shape[0], L))

  for i in range(mu.shape[0]):
    print('----------')
    print('[ITER] {:3d} mu = {:.4f}'.format(i, mu[i]))
    t_start = timeit.default_timer()
    models = h_step(train_features, codes, verbose=True)
    (A, old_Z) = f_step(train_features, models, verbose=True)
    (codes, loss) = z_step(train_features, models, A, old_Z, mu[i])
    t_end = timeit.default_timer()

    recon_error = np.linalg.norm(train_features-codes.dot(A), axis=1) ** 2
    print('[ITER] {:3d} train set recon error: {:.4f}'.format(i, sum(recon_error)/train_features.shape[0]))
    print('[ITER] {:3d} test  set recon error: {:.4f}'.format(i, test_recon(test_features, models, A)))
    print('[ITER] {:3d} {:.4f} seconds elapsed'.format(i, t_end-t_start))
    # print('[ITER] {:3d} loss: {:.4f}'.format(i, loss))

  for (m,i) in zip(models, range(len(models))):
    liblinearutil.save_model('models/tr{0:05d}-L{1:02d}-b{2:02d}.model'.format(train_features.shape[0], L, i), m)

  hash(features, num_train, L)
  print(calc_mean_ap(labels, num_test, num_train, L))
  print(calc_precision_at_k(labels, num_test, num_train, L, 50))
