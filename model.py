import numpy as np

def LinearRegression(x, y, lr):
    '''
    # Linear Regression
    ## Basic Concpet
    Model: linear
    Loss function: mean square error
    Opt_algo: SGD
    Learning_Rate: Static
    ## Attirbute
    x: traing data X
    y: traing data Y
    lr: learning rate
    '''
    bias = 0
    w = np.zeros(y.shape[1]) + 1 # initial weight: all 1
    for train_x in x:
        pass

def squre_error(y, y_pred):
    return (y-y_pred)**2

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def cal_loss(example_x, example_y):

def SGD(x, y, example_x, example_y, w, b):
    for i in range(x.shape[1]):
        w[i] = w[i] - 