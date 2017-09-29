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

    if type(lr) != (float or int):
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
    num_fea = x.shape[1]
    x_count = x.shape[0]
    for i in range(num_fea):
        cal_loss_w, cal_loss_b = cal_loss(x, y, b, w, i)
        # print('w:', cal_loss_w)
        # print('b :', cal_loss_b)  
        w[i] = w[i] - lr * cal_loss_w *(1 / x_count)
        b = b - lr * cal_loss_b * (1 / x_count)

    return b, w

def mse(y, y_pred):
    return sum((y-y_pred) ** 2) *(1 / y.shape[0])

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def cal_loss(x, y, b, w, i): 
    cal_weight = 0
    cal_bias = 0
    for j in range(x.shape[0]):
        cal_weight +=  y[j] - (b + np.dot(w, x[j]) )
        cal_bias += y[j] - (b + np.dot(w, x[j]))
    return -2 * cal_weight * w[i], -2 * cal_bias

def predcit(x, b, w):
    count = x.shape[0]
    y = np.array([])
    for i in range(count):
        y_const = b + np.dot(w, x[i])
        y = np.append(y, y_const)
    return y