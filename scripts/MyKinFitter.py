from dimension_magic import load_emg_data,get_glove_data,fast_ica_analysis
from PlotUtility import plot_multi_channel
import matplotlib.pyplot as plt
from group_stats import find_correlation
import numpy as np
from scipy.optimize import curve_fit
from signal_utilities import calculate_power_of_signal,custom_norm

def mykin_model(U,k01,k11,l01,l11,k02,k12,l02,l12):
    a1 = 1
    a2 = -1
    muscle_num = a1*(k01+k11*U[0])*(l01+l11*U[0])
    muscle_det = a1*(k01+k11*U[0])
    anti_num = a2*(k02+k12*U[1])*(l02+l12*U[1])
    anti_det = a2*(k02+k12*U[1])
    theta = (muscle_num+anti_num)/(muscle_det+anti_det)
    return theta

class MyoKinFitter:
    def __init__(self,joint_angle,joint_signals,muscle_signals,plot = True):
        self.joint_signals = joint_signals
        self.joint_angle = joint_angle
        self.muscle_signals = muscle_signals
        self.corr,self.stats = find_correlation(joint_angle,muscle_signals)
        self.njoint = joint_angle.shape[1]
        self.plot = plot

    def plot_correlation_coefficient_histogram_for_each_joint(self):
        plt.figure(figsize=[10,20])
        for i in range(self.njoint):
            plt.subplot(self.njoint,1,i+1)
            plt.title(f'finger {i}')
            plt.hist(self.corr[i],bins = 20)

    def plot_best_fit_channels(self):
        if self.plot:
            plt.figure(figsize=[10,40])
        self.fitting_ics = []
        for i in range(4):
            stati = self.stats[i]
            corri = self.corr[i][stati<0.005]
            maxid = np.argmax(corri)
            minid = np.argmin(corri)
            fingeri = self.joint_signals[:,i]
            poweri = self.muscle_signals[:,maxid]
            if self.plot:
                plt.subplot(12,1,3*(i)+1)
                plt.title(f'finger {i}')
                plt.plot(fingeri)
                plt.subplot(12,1,3*(i)+2)
                plt.title(f'power {i} r = {corri[maxid]}, p = {stati[stati<0.005][maxid]} id = {maxid}')
                plt.plot(poweri)
                plt.subplot(12,1,3*(i)+3)
                plt.title(f'power {i} r = {corri[minid]}, p = {stati[stati<0.005][minid]} id = {minid}')
                plt.plot(self.muscle_signals[:,minid])
            self.fitting_ics.append([maxid,minid])

    def fit_mykin_model(self):
        self.parameters = []
        for i in range(4):
            print(f'optimizing finger {i}')
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid]])
            try:
                popt, pcov = curve_fit(mykin_model, U*10,  self.joint_angle[:,i]*10,maxfev=1000,method = 'lm')
                self.parameters.append(popt)
            except RuntimeError:
                popt = np.ones(8)
                self.parameters.append(np.ones(8))

    def show_model_fit(self):
        for i in range(4):
            maxid,minid = self.fitting_ics[i]
            U = np.array([self.muscle_signals[:,maxid],self.muscle_signals[:,minid]])
            popt = self.parameters[i]
            theta_estimate = mykin_model(U,*popt)
            plt.figure(figsize=[10,5])
            plt.subplot(2,1,1)
            plt.plot(self.joint_signals[:,i])
            plt.subplot(2,1,2)
            plt.plot(theta_estimate)

# if __name__ == '__main__':
#     emg = load_emg_data()
#     emg_ica = fast_ica_analysis(emg)
#     emg_ica_power = calculate_power_of_signal(emg_ica)
#     plot_multi_channel(emg_ica_power)
#     glove = get_glove_data()
#     glove = custom_norm(glove)
#     glove = glove[:emg_ica_power.shape[0]]
#     fingers = glove[:,[4,7,11,15]]
#     fingers_power = calculate_power_of_signal(fingers)
#     fitter = MyoKinFitter(fingers,fingers_power,emg_ica_power)
#     fitter.plot_correlation_coefficient_histogram_for_each_joint()
#     fitter.plot_best_fit_channels()
#     fitter.fit_mykin_model()
#     fitter.show_model_fit()



