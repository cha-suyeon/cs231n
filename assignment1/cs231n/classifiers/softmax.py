# from builtins import range
# import numpy as np
# from random import shuffle
# from past.builtins import xrange


# def softmax_loss_naive(W, X, y, reg):
#     """
#     Softmax loss function, naive implementation (with loops)

#     Inputs have dimension D, there are C classes, and we operate on minibatches
#     of N examples.

#     Inputs:
#     - W: A numpy array of shape (D, C) containing weights.
#     - X: A numpy array of shape (N, D) containing a minibatch of data.
#     - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#       that X[i] has label c, where 0 <= c < C.
#     - reg: (float) regularization strength

#     Returns a tuple of:
#     - loss as single float
#     - gradient with respect to weights W; an array of same shape as W
#     """
#     # Initialize the loss and gradient to zero.
#     loss = 0.0
#     dW = np.zeros_like(W)

#     #############################################################################
#     # TODO: Compute the softmax loss and its gradient using explicit loops.     #
#     # Store the loss in loss and the gradient in dW. If you are not careful     #
#     # here, it is easy to run into numeric instability. Don't forget the        #
#     # regularization!                                                           #
#     #############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
#     # scores = X.dot(W)
#     # num_train = X.shape[0]
#     # num_classes = W.shape[1]

#     # # Softmax Loss
#     # for i in range(num_train):
#     #   softmax = np.exp(scores[i])/np.sum(np.exp(scores[i]))
#     #   loss += -np.log(softmax[y[i]])
#     #   # Weight Gradients
#     #   for j in range(num_classes):
#     #     dW[:,j] += X[i] * softmax[j]
#     #   dW[:,y[i]] -= X[i]

#     # # Average
#     # loss /= num_train
#     # dW /= num_train

#     # # Regularization
#     # loss += reg * np.sum(W * W)
#     # dW += reg * 2 * W 
    
#     num_classes = W.shape[1]  
#     num_train = X.shape[0]  
#     for i in range(num_train):
#         scores = X[i].dot(W)  
#         scores = scores - np.max(scores)  
#         scores_exp = np.exp(scores)     # 指数操作
#         ds_w = np.repeat(X[i], num_classes).reshape(-1, num_classes)   # 计算得分对权重的倒数
#         scores_exp_sum = np.sum(scores_exp)
#         pk = scores_exp[y[i]] / scores_exp_sum
#         loss += -np.log(pk)   
#         dl_s = np.zeros(W.shape)  # 开始计算loss对得分的倒数
#         for j in range(num_classes):
#             if j == y[i]:
#                 dl_s[:, j] = pk - 1    # 对于输入正确分类的那一项，倒数与其他不同
#             else:
#                 dl_s[:, j] = scores_exp[j] / scores_exp_sum
#         dW_i = ds_w * dl_s
#         dW += dW_i
#     loss /= num_train
#     dW /= num_train
#     loss += reg * np.sum(W * W)
#     dW += W * 2 * reg

    
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     return loss, dW


# def softmax_loss_vectorized(W, X, y, reg):
#     """
#     Softmax loss function, vectorized version.

#     Inputs and outputs are the same as softmax_loss_naive.
#     """
#     # Initialize the loss and gradient to zero.
#     loss = 0.0
#     dW = np.zeros_like(W)

#     #############################################################################
#     # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
#     # Store the loss in loss and the gradient in dW. If you are not careful     #
#     # here, it is easy to run into numeric instability. Don't forget the        #
#     # regularization!                                                           #
#     #############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     # scores = np.dot(X, W) # 500 x 10
#     # scores = np.exp(scores)
#     # scores = scores / np.sum(scores, axis = 1).reshape(-1, 1)
#     # loss = np.sum(-np.log(scores[np.arange(X.shape[0]), y])) / X.shape[0] + reg*np.sum(W*W)

#     # #dW
#     # scores[np.arange(X.shape[0]), y] -= 1 #not valid from here
#     # dW += np.dot(np.transpose(X), scores) / X.shape[0]  # 3072x10
#     # dW += 2*reg*W
    
#     num_classes = W.shape[1]
#     num_train = X.shape[0]
    
#     scores = X.dot(W)
#     scores = scores - np.max(scores, 1, keepdims=True)
#     scores_exp = np.exp(scores)
#     sum_s = np.sum(scores_exp, 1, keepdims=True)
#     p = scores_exp / sum_s
#     loss = np.sum(-np.log(p[np.arange(num_train), y]))

#     ind = np.zeros_like(p)
#     ind[np.arange(num_train), y] = 1
#     dW = X.T.dot(p - ind)

#     loss /= num_train
#     dW /= num_train
#     loss += reg * np.sum(W * W)
#     dW += W * 2 * reg
    
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     return loss, dW

from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]  
    num_train = X.shape[0] 
    for i in range(num_train):
        scores = X[i].dot(W)  
        scores = scores - np.max(scores)  
        scores_exp = np.exp(scores)     # 指数操作
        ds_w = np.repeat(X[i], num_classes).reshape(-1, num_classes)   # 计算得分对权重的倒数
        scores_exp_sum = np.sum(scores_exp)
        pk = scores_exp[y[i]] / scores_exp_sum
        loss += -np.log(pk)   
        dl_s = np.zeros(W.shape)  # 开始计算loss对得分的倒数
        for j in range(num_classes):
            if j == y[i]:
                dl_s[:, j] = pk - 1    # 对于输入正确分类的那一项，倒数与其他不同
            else:
                dl_s[:, j] = scores_exp[j] / scores_exp_sum
        dW_i = ds_w * dl_s
        dW += dW_i
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += W * 2 * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W)
    scores = scores - np.max(scores, 1, keepdims=True)
    scores_exp = np.exp(scores)
    sum_s = np.sum(scores_exp, 1, keepdims=True)
    p = scores_exp / sum_s
    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += W * 2 * reg

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW