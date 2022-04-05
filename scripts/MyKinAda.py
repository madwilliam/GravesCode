import matplotlib.pyplot as plt
from group_stats import find_correlation,custom_correlation
import numpy as np
from random import sample
import math
from scipy.optimize import curve_fit,minimize,approx_fprime

class MyKinAda:
    def __init__(self,joint_angle,muscle_signals,plot = True,correlation_function = find_correlation,derivative = True):
        self.joint_angle = joint_angle
        self.muscle_signals = muscle_signals
        self.joint_angle_extension = self.get_extension_signals(self.joint_angle,derivative)
        self.joint_angle_flexion = self.get_flexion_signals(self.joint_angle,derivative)
        self.extension_corr,self.extension_stats = correlation_function(self.joint_angle_extension,muscle_signals)
        self.flexion_corr,self.flexion_stats = correlation_function(self.joint_angle_flexion,muscle_signals)
        self.njoint = joint_angle.shape[1]
        self.plot = plot
        
    def mykin_model(self,x,U):
        a1,a2,k01,k11,l01,l11,k02,k12,l02,l12 = np.abs(x)
        a2 = -a2
        muscle_num = a1*(k01+k11*U[0])*(l01+l11*U[0])
        muscle_det = a1*(k01+k11*U[0])
        anti_num = a2*(k02+k12*U[1])*(l02+l12*U[1])
        anti_det = a2*(k02+k12*U[1])
        theta = (muscle_num+anti_num)/(muscle_det+anti_det)
        return theta
    
    def objective(self,x,U):
        U = np.array(U).T
        theta = self.mykin_model(x,U[:2])
        return np.sqrt(np.sum(np.square(theta-U[2])))

    def derivative(self,x,U):
        eps = np.sqrt(np.finfo(float).eps)
        grad = approx_fprime(x, self.objective, [eps for _ in range(len(x))], U)
        return grad

    def adagrad(self,x0,data,stepsize = 1e-2,fudge_factor = 1e-6,max_it=5000,minibatchsize=None,minibatch_ratio=0.01,atol = 1e-8,**kwargs):
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
        errors = np.zeros(max_it)
        for i in range(max_it):
            s=sample(range(ld),minibatchsize)
            sd=[data[idx] for idx in s]
            grad=self.derivative(w,sd,**kwargs)
            gti+=grad**2
            adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
            w = w - stepsize*adjusted_grad
            errors[i] = self.objective(w,data,**kwargs)
            if np.abs(errors[i-1]-errors[i])<atol:
                break
        plt.figure()
        plt.plot(errors)
        plt.xlabel = 'iterations'
        plt.ylabel = 'RMSE'
        plt.title(self.title)
        return w

    def get_extension_signals(self,signals,derivative):
        if derivative:
            signals = np.pad(signals[1:]-signals[:-1],[0,1])
        extension_signals = np.zeros(signals.shape)
        ispositive = signals>0
        extension_signals[ispositive] = signals[ispositive]
        return extension_signals
    
    def get_flexion_signals(self,signals,derivative):
        if derivative:
            signals = np.pad(signals[1:]-signals[:-1],[0,1])
        flexion_signals = np.zeros(signals.shape)
        is_negative = signals<0
        flexion_signals[is_negative] = signals[is_negative]
        return -flexion_signals

    def plot_correlation_coefficient_histogram_for_each_joint(self):
        plt.figure(figsize=[10,20])
        for i in range(self.njoint):
            plt.subplot(self.njoint,1,i+1)
            plt.title(f'finger {i} extension')
            plt.hist(self.extension_corr[i],bins = 20)
        plt.figure(figsize=[10,20])
        for i in range(self.njoint):
            plt.subplot(self.njoint,1,i+1)
            plt.title(f'finger {i} flexion')
            plt.hist(self.extension_corr[i],bins = 20)

    def plot_best_fit_channels(self,alpha = 0.005):
        if self.plot:
            plt.figure(figsize=[10,60])
        self.fitting_ics = []
        for i in range(4):
            stati_extension = self.extension_stats[i]
            corri_extension = self.extension_corr[i][stati_extension<alpha]
            maxid_extension = np.argmax(np.abs(corri_extension))
            stati_flexion = self.flexion_stats[i]
            corri_flexion = self.flexion_corr[i][stati_flexion<alpha]
            maxid_flexion = np.argmax(np.abs(corri_flexion))
            if self.plot:
                plt.subplot(16,1,4*(i)+1)
                plt.title(f'finger {i} extension')
                plt.plot(self.joint_angle_extension[:,i])
                plt.subplot(16,1,4*(i)+2)
                r_extension = corri_extension[maxid_extension]
                r_flexion = corri_extension[maxid_extension]
                p_extension = stati_flexion[stati_flexion<alpha][maxid_flexion]
                p_flexion = stati_flexion[stati_flexion<alpha][maxid_flexion]
                plt.title(f'Extension power {i} r = {r_extension}, p = {p_extension} id = {maxid_extension}')
                plt.plot(self.muscle_signals[:,maxid_extension])
                plt.subplot(16,1,4*(i)+3)
                plt.title(f'finger {i} flexion')
                plt.plot(self.joint_angle_flexion[:,i])
                plt.subplot(16,1,4*(i)+4)
                plt.title(f'Flexion power {i} r = {r_flexion}, p = {p_flexion} id = {maxid_flexion}')
                plt.plot(self.muscle_signals[:,maxid_flexion])
            self.fitting_ics.append([maxid_extension,maxid_flexion])

    def fit_mykin_model(self,*args, **kwargs):
        self.parameters = []
        for i in range(4):
            print(f'optimizing finger {i}')
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid],self.joint_angle[:,i]])
            try:
                self.title = f'finger {i}'
                res = self.adagrad(x0=np.array([1,-1,1,1,1,1,1,1,1,1]),data = U.T)
                self.parameters.append(res)
            except RuntimeError:
                popt = np.ones(10)
                self.parameters.append(np.ones(10))

    def show_model_fit(self):
        for i in range(4):
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid]])
            w = self.parameters[i]
            theta_estimate = self.mykin_model(w,U)
            plt.figure(figsize=[10,5])
            plt.subplot(2,1,1)
            plt.plot(self.joint_angle[:,i])
            plt.subplot(2,1,2)
            plt.plot(theta_estimate)