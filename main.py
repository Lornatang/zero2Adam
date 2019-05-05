""" Main func.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

from model import dnn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_data, y_data = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.8)

X_train = X_train.T
y_train = y_train.reshape(y_train.shape[0], -1).T

X_test = X_test.T
y_test = y_test.reshape(y_test.shape[0], -1).T

accuracy = dnn(X_train,
               y_train,
               X_test,
               y_test,
               layer_dims=[X_train.shape[0], 10, 5, 1],
               learning_rate=0.001,
               num_iterations=10000)

print(f"Acc: {accuracy}")
