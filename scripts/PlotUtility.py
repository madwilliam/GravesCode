import matplotlib.pyplot as plt

def plot_multi_channel(data,x = [],titles = []):
    if data.shape[0] < data.shape[1]:
        data = data.T
    nplot = data.shape[1]
    plt.figure(figsize = [10,5*nplot])
    for i in range(nplot):
        plt.subplot(nplot,1,i+1)
        if len(x)>0:
            plt.plot(x,data[:,i])
        else:
            plt.plot(data[:,i])
        if len(titles)>0:
            plt.title(titles[i])
