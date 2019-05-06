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


def linear_backward(dx, cache):
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
  x, W, _, _, _, _, _ = cache
  dW = np.dot(dx, x.T)
  db = np.sum(dx, axis=1, keepdims=True)
  dx = np.dot(W.T, dx)

  return dx, dW, db


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


def relu_backward(dx, x):
  """ derivation of relu
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return np.multiply(dx, np.int64(x > 0))


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


def batch_norm(x, gamma, beta, epsilon=1e-12):
  """
  Paras
  -----------------------------------
  x:       the input of activation (x = np.dot(x, W) + b)
  gamma:   zoom factor
  beta:    translation factor
  epsilon: is a constant for denominator is 0

  Returns
  -----------------------------------
  z_out, mean, variance
  """
  mean = np.mean(x, axis=1, keepdims=True)  # cal x mean
  var = np.var(x, axis=1, keepdims=True)  # cal x var
  sqrt_var = np.sqrt(var + epsilon)

  normalized = (x - mean) / sqrt_var  # normalized

  # scale and shift variables are introduced to calculate the normalized value
  out = np.multiply(gamma, normalized) + beta
  return mean, var, sqrt_var, normalized, out


def batch_norm_backward(dx, cache):
  """ derivation of batch_norm
  Paras
  -----------------------------------
  dx: output of the linear layer

  Returns
  -----------------------------------
  """
  _, _, _, gamma, sqrt_var, normalized, _ = cache
  m = dx.shape[1]
  dgamma = np.sum(dx * normalized, axis=1, keepdims=True)
  dbeta = np.sum(dx, axis=1, keepdims=True)
  dout = 1. / m * gamma * sqrt_var * (
              m * dx - np.sum(dx, axis=1, keepdims=True) - normalized * np.sum(dx * normalized, axis=1, keepdims=True))
  return dgamma, dbeta, dout
