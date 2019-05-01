"""Implement some basic operations of Activation function.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

import numpy as np


def linear(x, w, b):
  """ linear activation func
  Paras
  -----------------------------------
  x: input of the linear layer
  w: weight
  b: bias

  Returns
  -----------------------------------
  cal wx + b
  """

  return np.dot(w, x) + b


def linear_backward(x, cache):
  """
  Paras
  -----------------------------------
  x: input of the linear layer
  w: weight
  b: bias

  Returns
  -----------------------------------
  cal wx + b
  """
  A, W, b, z = cache
  dw = np.dot(x, A.T)
  db = np.sum(x, axis=1, keepdims=True)
  dx = np.dot(W.T, x)

  return dx, dw, db


def relu(x):
  """ relu activation function
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return np.maximum(0, x)


def relu_backward(x):
  """ derivation of relu
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return np.multiply(1., np.int64(x > 0))


def sigmoid(x):
  """ implement the activation function sigmoid
  Paras
  -----------------------------------
  x: Output of the linear layer

  Returns
  -----------------------------------
  max of nums"""

  return 1 / (1 + np.exp(-x))


def sigmoid_backward(x):
  """ derivation of sigmoid
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """
  s = sigmoid(x)

  return s * (1 - s)


def tanh(x):
  """ implement the activation function tanh
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return (1 - np.exp(2 * -x)) / (1 + np.exp(2 * -x))


def tanh_backward(x):
  """derivation of tanh
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """
  favl = tanh(x)

  return 1 - favl ** 2
