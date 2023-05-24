from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # loss function의 gradient인 dW를 구하는 것 (derivative of weights)
    dW /= num_train # 1/N - scale gradient over the number of samples
    dW = dW + 2 * reg * W # append partial derivative of regularization term (use L2 norm so multiply 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1] # 10 - (3073, 10)
    num_train = X.shape[0] # 500 - (500,)
    
    scores = X.dot(W) # 행렬 곱
    # scores for true labels
    correct_class_score = scores[ np.arange(num_train), y].reshape(num_train,1) # scores[500, 500].reshape(-1, 1) = (250000, 1) ?
    # margin for scores (for문을 통해 하나씩 비교했지만 행렬 형태라서 간단하게 가능)
    margins = np.maximum(0, scores - correct_class_score + 1)
    margins[np.arange(num_train), y] = 0 # do not consider correct class in loss
    # maximum의 경우, element-wise maximum of array elements. (max와 다름)
    # 즉, 각 배열들끼리 비교해서 그 요소들 중 max 값이 저장됨
    # margins.sum() -> 모두 더하면 loss 값이 됨
    loss = margins.sum() / num_train # svm loss 
    loss += reg * np.sum(W*W)  # regularized loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1 # 1의 갯수 = loop 도는 횟수 (1의 갯수만큼 )
    valid_margin_count = margins.sum(axis=1) 
    # Subtract in correct class (-s_y)
    margins[np.arange(num_train),y ] -= valid_margin_count # Subtract in correct class (- s_j)
    dW = (X.T).dot(margins) / num_train # 내적한 값에 1/N을 곱함

    # Regularization gradient
    dW = dW + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
