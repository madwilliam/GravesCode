import numpy as np
from random import sample
import math
import numpy as np
from scipy import optimize
global U

def mykin_model(x,U):
    k01,k11,l01,l11,k02,k12,l02,l12 = x 
    a1 = 1
    a2 = -1
    muscle_num = a1*(k01+k11*U[0])*(l01+l11*U[0])
    muscle_det = a1*(k01+k11*U[0])
    anti_num = a2*(k02+k12*U[1])*(l02+l12*U[1])
    anti_det = a2*(k02+k12*U[1])
    theta = (muscle_num+anti_num)/(muscle_det+anti_det)
    return theta
 
def derivative(x,U):
    eps = np.sqrt(np.finfo(float).eps)
    grad = optimize.approx_fprime(x, mykin_model, [eps for _ in range(len(x))], U)
    return grad

def adagrad(f_grad,x0,data,args,stepsize = 1e-2,fudge_factor = 1e-6,max_it=1000,minibatchsize=None,minibatch_ratio=0.01):
    # f_grad returns the loss functions gradient
    # x0 are the initial parameters (a starting point for the optimization)
    # data is a list of training data
    # args is a list or tuple of additional arguments passed to fgrad
    # stepsize is the global stepsize for adagrad
    # fudge_factor is a small number to counter numerical instabiltiy
    # max_it is the number of iterations adagrad will run
    # minibatchsize if given is the number of training samples considered in each iteration
    # minibatch_ratio if minibatchsize is not set this ratio will be used to determine the batch size dependent on the length of the training data
    
    #d-dimensional vector representing diag(Gt) to store a running total of the squares of the gradients.
    gti=np.zeros(x0.shape[0])
    
    ld=len(data)
    if minibatchsize is None:
        minibatchsize = int(math.ceil(len(data)*minibatch_ratio))
    w=x0
    for t in range(max_it):
        s=sample(range(ld),minibatchsize)
        sd=[data[idx] for idx in s]
        grad=f_grad(w,sd,*args)
        gti+=grad**2
        adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
        w = w - stepsize*adjusted_grad
    return w