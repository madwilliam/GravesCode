import matplotlib.pyplot as plt
from group_stats import find_correlation,custom_correlation
import numpy as np
from scipy.optimize import curve_fit,minimize
def mykin_model(U,a1,a2,k01,k11,l01,l11,k02,k12,l02,l12):
    k01,k11,l01,l11,k02,k12,l02,l12 = np.abs([k01,k11,l01,l11,k02,k12,l02,l12])
    a1 = 1
    a2 = -1
    muscle_num = a1*(k01+k11*U[0])*(l01+l11*U[0])
    muscle_det = a1*(k01+k11*U[0])
    anti_num = a2*(k02+k12*U[1])*(l02+l12*U[1])
    anti_det = a2*(k02+k12*U[1])
    theta = (muscle_num+anti_num)/(muscle_det+anti_det)
    return theta

class MyKinPro:
    def __init__(self,joint_angle,muscle_signals,plot = True,correlation_function = find_correlation):
        self.joint_angle = joint_angle
        self.muscle_signals = muscle_signals
        self.joint_angle_extension = self.get_extension_signals(self.joint_angle)
        self.joint_angle_flexion = self.get_flexion_signals(self.joint_angle)
        self.extension_corr,self.extension_stats = correlation_function(self.joint_angle_extension,muscle_signals)
        self.flexion_corr,self.flexion_stats = correlation_function(self.joint_angle_flexion,muscle_signals)
        self.njoint = joint_angle.shape[1]
        self.plot = plot
        
    def get_extension_signals(self,signals):
        extension_signals = np.zeros(signals.shape)
        ispositive = signals>0
        extension_signals[ispositive] = signals[ispositive]
        return extension_signals
    
    def get_flexion_signals(self,signals):
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

    def fit_mykin_model(self,method = 'nelder-mead',options = {'xatol': 1e-8, 'disp': True,'maxiter' : 5000}):
        self.parameters = []
        for i in range(4):
            print(f'optimizing finger {i}')
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid]])
            def mykin_model(X):
                a1,a2,k01,k11,l01,l11,k02,k12,l02,l12 = X
                muscle_num = a1*(k01+k11*U[0])*(l01+l11*U[0])
                muscle_det = a1*(k01+k11*U[0])
                anti_num = a2*(k02+k12*U[1])*(l02+l12*U[1])
                anti_det = a2*(k02+k12*U[1])
                theta = (muscle_num+anti_num)/(muscle_det+anti_det)
                return np.sqrt(np.sum(np.square(theta-self.joint_angle[:,i])))
            try:
                res = minimize(mykin_model,x0 =   np.array([1,-1,1,1,1,1,1,1,1,1]), method=method,
               options=options,bounds = [(0,10),(-10,0),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10),(0,10)])
                self.parameters.append(res.x)
            except RuntimeError:
                popt = np.ones(10)
                self.parameters.append(np.ones(10))

    def show_model_fit(self):
        for i in range(4):
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid]])
            popt = self.parameters[i]
            theta_estimate = mykin_model(U,*popt)
            plt.figure(figsize=[10,5])
            plt.subplot(2,1,1)
            plt.plot(self.joint_angle[:,i])
            plt.subplot(2,1,2)
            plt.plot(theta_estimate)