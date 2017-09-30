import numpy as np
from numpy.linalg import inv
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

    if type(lr) == str:
        print('目前只有 static learning rate\n 使用 lr = 0.0001')
    else:
        # 數次 epoch 的 SGD, 還沒做 random choice
        for i in range(epoch):
            b, w = SGD(x, y, lr, b, w)

    return b, w

def LinearRegression_close(x, y):
    '''
    # 用於驗證
    ## Return
    b: bias
    w: weight array
    '''
    x = np.insert(x, 0, values = 1, axis = 1)
    x_trans = np.transpose(x)
    
    y_mat = np.transpose(np.mat(y))
    # print(y_mat.shape)
    x_mat = np.mat(x)
    # print(x_mat.shape)
    x_trans_mat = np.mat(x_trans)
    # print(x_trans_mat.shape)

    w = inv(x_trans_mat * x_mat)  * x_trans_mat * y_mat
    w = np.array(w)
    w = w.flatten()
    # print(w)

    return w[0], w[1:] #return b, w

def SGD(x, y, lr, b, w):
    '''
    ## Attribute
    x: example x
    y: example y
    lr: learning rates 
    b: constant bias
    w: weight array

    ## Return 
    b: constant bias
    w: weight array
    '''
    num_fea = x.shape[1]
    x_count = x.shape[0]
    for i in range(num_fea):
        gradient_w, gradient_b = gradient(x, y, b, w, i)
        # print('w:', gradient_w)
        # print('b :', gradient_b)  
        w[i] = w[i] - lr * gradient_w *(1 / x_count)
        b = b - lr * gradient_b * (1 / x_count)

    return b, w

<<<<<<< HEAD
def mse(y, y_pred):
    return sum((y-y_pred) ** 2) *(1 / y.shape[0])

def y_pred_linear(x, bias, w):
    return bias + np.dot(w, x)

def gradient(x, y, b, w, i): 
=======
def cal_loss(x, y, b, w, i):
>>>>>>> dev
    cal_weight = 0
    cal_bias = 0
    for j in range(x.shape[0]):
        cal_weight +=  (y[j] - (b + np.dot(w, x[j]))) * x[j, i]
        cal_bias += y[j] - (b + np.dot(w, x[j]))
    return -2 * cal_weight , -2 * cal_bias

def predcit(x, b, w):
    count = x.shape[0]
    y = np.array([])
    for i in range(count):
        y_const = b + np.dot(w, x[i])
        y = np.append(y, y_const)
    return y

def mse(y, y_pred):
    return sum((y-y_pred) ** 2) *(1 / y.shape[0])

def adagrad(lr, se, run):
    pass