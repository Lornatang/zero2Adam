"""Implement some basic operations of SGD.
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
  batches:    list of synchronous (data, mini_batch_Y)
  """
  m = data.shape[1]  # number of training examples
  batches = []

  # Step 1: Shuffle (data, label)
  permutation = list(np.random.permutation(m))
  data = data[:, permutation]
  label = label[:, permutation].reshape((1, m))

  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  # number of mini batches of size mini_batch_size in your partitioning
  num_batches = m // batch_size
  for k in range(0, num_batches):
    data = data[:, k * batch_size: (k + 1) * batch_size]
    label = label[:, k * batch_size: (k + 1) * batch_size]
    batch = (data, label)
    batches.append(batch)

  # Handling the end case (last mini-batch < mini_batch_size)
  if m % batch_size != 0:
    data = data[:, num_batches * batch_size: m]
    label = label[:, num_batches * batch_size: m]
    batch = (data, label)
    batches.append(batch)

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
    paras["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
      2 / layer_dims[l - 1])  # he initialization
    paras["b" + str(l)] = np.zeros((layer_dims[l], 1))

  return paras


def initialize_adam(paras):
  """ Initializes v and s as two python dictionaries with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
  Paras
  -----------------------------------
  parameters:   python dictionary containing your parameters.
                parameters["W" + str(l)] = Wl
                parameters["b" + str(l)] = bl

  Returns
  -----------------------------------
  v:            python dictionary that will contain the exponentially weighted average of the gradient.
                v["dW" + str(l)] = ...
                v["db" + str(l)] = ...
  s:            python dictionary that will contain the exponentially weighted average of the squared gradient.
                s["dW" + str(l)] = ...
                s["db" + str(l)] = ...
  """
  L = len(paras) // 2  # number of layers in the neural networks
  v = {}
  s = {}
  # Initialize v, s. Input: "parameters". Outputs: "v, s".
  for l in range(L):
    v["dW" + str(l + 1)] = np.zeros(paras["W" + str(l + 1)].shape)
    v["db" + str(l + 1)] = np.zeros(paras["b" + str(l + 1)].shape)
    s["dW" + str(l + 1)] = np.zeros(paras["W" + str(l + 1)].shape)
    s["db" + str(l + 1)] = np.zeros(paras["b" + str(l + 1)].shape)

  return v, s


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
  A = x
  caches = [(None, None, None, x)]
  # calculate from 1 to L-1 layer
  for l in range(1, L):
    A_pre = A
    W = paras["W" + str(l)]
    b = paras["b" + str(l)]
    z = np.dot(W, A_pre) + b  # 计算z = wx + b
    A = relu(z)  # relu activation function
    caches.append((W, b, z, A))
  # calculate Lth layer
  WL = paras["W" + str(L)]
  bL = paras["b" + str(L)]
  zL = np.dot(WL, A) + bL
  AL = sigmoid(zL)
  caches.append((WL, bL, zL, AL))
  return AL, caches


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
  # print("L:   " + str(L))
  # calculate the Lth layer gradients
  prev_AL = caches[L - 1][3]
  dzL = 1. / m * (pred - label)
  # print(dzL.shape)
  # print(prev_AL.T.shape)
  dWL = np.dot(dzL, prev_AL.T)
  dbL = np.sum(dzL, axis=1, keepdims=True)
  gradients = {"dW" + str(L): dWL, "db" + str(L): dbL}
  # calculate from L-1 to 1 layer gradients
  for l in reversed(range(1, L)):  # L-1,L-3,....,1
    post_W = caches[l + 1][0]  # 要用后一层的W
    dz = dzL  # 用后一层的dz

    dal = np.dot(post_W.T, dz)
    z = caches[l][2]  # 当前层的z
    dzl = np.multiply(dal, relu_backward(z))
    prev_A = caches[l - 1][3]  # 前一层的A
    dWl = np.dot(dzl, prev_A.T)
    dbl = np.sum(dzl, axis=1, keepdims=True)

    gradients["dW" + str(l)] = dWl
    gradients["db" + str(l)] = dbl
    dzL = dzl  # 更新dz
  return gradients


def compute_loss(pred, label):
  """ calculate loss function
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


def update_parameters_with_adam(paras, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
  """ Update parameters using Adam
  Paras
  ------------------------------------
  parameters:     python dictionary containing your parameters:
                  parameters['W' + str(l)] = Wl
                  parameters['b' + str(l)] = bl
  grads:          python dictionary containing your gradients for each parameters:
                  grads['dW' + str(l)] = dWl
                  grads['db' + str(l)] = dbl
  v:              Adam variable, moving average of the first gradient, python dictionary
  s:              Adam variable, moving average of the squared gradient, python dictionary
  learning_rate:  the learning rate, scalar.
  beta1:          Exponential decay hyperparameter for the first moment estimates
  beta2:          Exponential decay hyperparameter for the second moment estimates
  epsilon:        hyperparameter preventing division by zero in Adam updates

  Returns
  ------------------------------------
  parameters:     python dictionary containing your updated parameters
  """

  L = len(paras) // 2  # number of layers in the neural networks
  v_corrected = {}  # Initializing first moment estimate, python dictionary
  s_corrected = {}  # Initializing second moment estimate, python dictionary

  # Perform Adam update on all parameters
  for l in range(L):
    # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
    v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
    v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
    # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
    v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
    v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
    # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
    s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
    s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
    # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
    s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
    s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
    # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
    paras["W" + str(l + 1)] = paras["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
    paras["b" + str(l + 1)] = paras["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

  return paras
