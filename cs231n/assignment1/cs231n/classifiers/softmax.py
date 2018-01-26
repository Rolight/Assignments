import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  for idx in range(N):
      cur_x = np.reshape(X[idx, :], (1, -1))
      cur_y = y[idx]
      cur_result = np.matmul(cur_x, W)
      cur_result -= np.max(cur_result)
      cur_result = np.exp(cur_result)
      # calc softmax
      cur_result = cur_result / np.sum(cur_result)
      loss += -np.log(cur_result[0, cur_y])
      cur_result[0, cur_y] -= 1
      dW += np.matmul(cur_x.T, cur_result)
  loss /= N
  loss += reg * np.sum(W * W) * 0.5
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  result = np.matmul(X, W)
  result -= np.max(result, axis=1, keepdims=True)
  result = np.exp(result) / np.sum(np.exp(result), axis=1, keepdims=True)
  N = X.shape[0]
  loss = -np.sum(np.log(result[np.arange(N), y]))
  loss /= N
  loss += reg * np.sum(W * W) * 0.5
  result[np.arange(N), y] -= 1
  dW = np.matmul(X.T, result)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

