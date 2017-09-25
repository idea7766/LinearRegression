import numpy as np

def LinearRegression(x, y, lr = 0.0001 , epoch = 5, ex_size = 20):
    '''
    # Linear Regression
    ## Basic Concpet
    model: Linear_function 
    Loss_function: square error
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
    b = 0
    w = np.ones(x.shape[1])

    if type(lr) != float:
        print('目前只有 static learning rate\n 使用 lr = 0.0001')

    # 數次 epoch 的 SGD, 還沒做 random choice
    for i in range(epoch):
        b, w = SGD(x, y, lr, b, w)

    return b, w

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
        # cal_loss_w = float(cal_loss_w)
        # cal_loss_b = float(cal_loss_b)
        w[i] = w[i] - lr * cal_loss_w
        b = b - lr * cal_loss_b
    return b, w

def squre_error(y, y_pred):
    return (y-y_pred)**2

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def cal_loss(x, y, b, w, i): 
    cal_weight = np.zeros(x.shape[1])
    cal_bias = 0
    # print(w.shape)
    # print(x.shape)
    for j in range(x.shape[0]):
        cal_weight +=  y[j] - (b + np.dot(w, x[j]) )
        cal_bias += y[j] - (b + np.dot(w, x[j]))
    # cal_error = -2 * (cal_weight + cal_bias)
    print(cal_weight.shape)
    return -2 * cal_weight * w[i], -2 * cal_bias
