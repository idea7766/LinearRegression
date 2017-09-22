import numpy as np

def LinearRegression(x, y, lr, epoch):
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
    epoch: epoch數
    '''
    if x.ndim !=2:
        raise('= =寫錯了啦幹')
    bias = 0
    w = np.ones(y.shape[1]) # initial weight: all 1

    for train_x in x:
        pass
    return b, w

def squre_error(y, y_pred):
    return (y-y_pred)**2

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def cal_loss(x, y, b, w):
    return (y[i] - (b + dot(w * x[i]) ) * w[i])

def SGD(x, y, lr, b, w):
    '''
    # Attribute
    x: example x
    y: example y
    lr: learning rates 
    # Return 
    b: constant bias
    w: weight array
    '''
    w_result = np.array(x.shape)
    b_result = 
    for i in range(x.shape[1]):
        w_result[i] = w[i] - lr * 
        return b, w