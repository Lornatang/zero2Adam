"""Implement some basic operations of Adam.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

from activation import *


def random_mini_batches(data, label, batch_size):
  """ creates a list of random mini batches from (data, label)
  Paras
  -----------------------------------
  data:            input data, of shape (input size, number of examples)
  label:           true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  mini_batch_size: size of the mini-batches, integer

  Returns
  -----------------------------------
  batches:    list of synchronous (mini_batch_X, mini_batch_Y)
  """
  m = data.shape[1]  # number of training examples
  batches = []

  # Step 1: Shuffle (data, label)
  permutation = list(np.random.permutation(m))
  shuffled_X = data[:, permutation]
  shuffled_Y = label[:, permutation].reshape((1, m))

  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  # number of mini batches of size mini_batch_size in your partitioning
  num_batches = m // batch_size
  for k in range(0, num_batches):
    mini_batch_X = shuffled_X[:, k * batch_size: (k + 1) * batch_size]
    mini_batch_Y = shuffled_Y[:, k * batch_size: (k + 1) * batch_size]
    mini_batch = (mini_batch_X, mini_batch_Y)
    batches.append(mini_batch)

  # Handling the end case (last mini-batch < mini_batch_size)
  if m % batch_size != 0:
    mini_batch_X = shuffled_X[:, num_batches * batch_size: m]
    mini_batch_Y = shuffled_Y[:, num_batches * batch_size: m]
    mini_batch = (mini_batch_X, mini_batch_Y)
    batches.append(mini_batch)

  return batches


def init_paras(layer_dims):
  """ initial paras ops
  Paras
  -----------------------------------
  layer_dims: list, the number of units in each layer (dimension)

  Returns
  -----------------------------------
  dictionary: storage parameters w1,w2...wL, b1,...bL
  """
  L = len(layer_dims)
  paras = {}
  for l in range(1, L):
    paras["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
    paras["b" + str(l)] = np.zeros((layer_dims[l], 1))

  return paras


def forward_propagation(x, paras):
  """ forward propagation function
  Paras
  ------------------------------------
  x:          input dataset, of shape (input size, number of examples)

  parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2",...,"WL", "bL"
              W -- weight matrix of shape (size of current layer, size of previous layer)
              b -- bias vector of shape (size of current layer,1)

  Returns
  ------------------------------------
  y:          the output of the last Layer(y_predict)
  caches:     list, every element is a tuple:(W,b,z,A_pre)
  """
  L = len(paras) // 2  # number of layer
  # A = x
  caches = []
  # calculate from 1 to L-1 layer
  for l in range(1, L):
    w = paras["W" + str(l)]
    b = paras["b" + str(l)]

    # linear forward -> relu forward ->linear forward....
    z = linear(x, w, b)
    caches.append((x, w, b, z))
    x = relu(z)

  # calculate Lth layer
  w = paras["W" + str(L)]
  b = paras["b" + str(L)]

  z = linear(x, w, b)
  caches.append((x, w, b, z))
  y = sigmoid(z)

  return y, caches


def backward_propagation(pred, label, caches):
  """ implement the backward propagation presented.
  Paras
  ------------------------------------
  pred:   pred "label" vector (containing 0 if cat, 1 if non-cat)
  label:  true "label" vector (containing 0 if cat, 1 if non-cat)
  caches: caches output from forward_propagation(),(W,b,z,pre_A)

  Returns
  ------------------------------------
  gradients -- A dictionary with the gradients with respect to dW,db
  """
  m = label.shape[1]
  L = len(caches) - 1

  # calculate the Lth layer gradients
  z = 1. / m * (pred - label)

  _, w, b = linear_backward(z, caches[L])
  gradients = {"dW" + str(L + 1): w, "db" + str(L + 1): b}

  # calculate from L-1 to 1 layer gradients
  for l in reversed(range(0, L)):  # L-1,L-3,....,1
    _, _, _, z = caches[l]
    # ReLu backward -> linear backward
    # relu backward
    out = relu_backward(z)
    # linear backward
    _, w, b = linear_backward(out, caches[l])

    gradients["dW" + str(l + 1)] = w
    gradients["db" + str(l + 1)] = b

  return gradients


def compute_loss(pred, label):
  """calculate loss function
  Paras
  ------------------------------------
  pred:  pred "label" vector (containing 0 if cat, 1 if non-cat)

  label: true "label" vector (containing 0 if cat, 1 if non-cat)

  Returns
  ------------------------------------
  loss:  the difference between the true and predicted values
  """
  loss = 1. / label.shape[1] * np.nansum(np.multiply(-np.log(pred), label) + np.multiply(-np.log(1 - pred), 1 - label))

  return np.squeeze(loss)


def update_parameters_with_sgd(paras, grads, learning_rate):
  """ update parameters using SGD
  ```
	VdW = beta * VdW - learning_rate * dW
	Vdb = beta * Vdb - learning_rate * db
	W = W + beta * VdW - learning_rate * dW
	b = b + beta * Vdb - learning_rate * db
	```
  Paras
  ------------------------------------
  parameters:    python dictionary containing your parameters:
                 parameters['W' + str(l)] = Wl
                 parameters['b' + str(l)] = bl
  grads:         python dictionary containing your gradients for each parameters:
                 grads['dW' + str(l)] = dWl
                 grads['db' + str(l)] = dbl
  learning_rate: the learning rate, scalar.

  Returns
  ------------------------------------
  parameters:     python dictionary containing your updated parameters

  """
  L = len(paras) // 2
  for l in range(L):
    paras["W" + str(l + 1)] = paras["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
    paras["b" + str(l + 1)] = paras["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

  return paras
