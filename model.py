import numpy as np

def LinearRegression(x, y, lr, epoch = 5):
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
        raise('= =寫二維陣列啦')
    
    # initialization
    bias = 0
    w = np.ones(y.shape[1])

    # 數次 epoch 的 SGD
    for i in range(epoch):
        SGD()
        
    for train_x in x:
        pass
    return b, w

def squre_error(y, y_pred):
    return (y-y_pred)**2

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def cal_loss(x, y, b, w, i): 
    for j in range(x.shape[0]):
        cal_weight +=  y[j] - (b + dot(w, x[j]) )
        cal_bias += (y[j] - (b + dot(w, x[j]) )
    # cal_error = -2 * (cal_weight + cal_bias)
    return -2 * cal_weight * w[i], -2 * cal_bias

def SGD(x, y, lr, b, w):
    '''
    ## Attribute
    x: example x
    y: example y
    lr: learning rates 
    ## Return 
    b: constant bias
    w: weight array
    '''
    # w_result = np.array(x.shape)
    # b_result = 0
    for i in range(x.shape[1]):
        cal_loss_w, cal_loss_b = cal_loss(x, y, b, w, i)
        w[i] = w[i] - lr * cal_loss_w
        b = b - lr * cal_loss_b
    return b, w