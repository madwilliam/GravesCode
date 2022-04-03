from sklearn.decomposition import PCA
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
io = FileIO()
exps = io.load_experiment('s1')
emg = exps['S1_E1_A1']['emg']
pca = PCA(n_components=10, svd_solver='full')
pca.fit(emg)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
t = pca.transform(emg)
fig = go.Figure(data=go.Scatter3d(
    x=t[:1000,0], y=t[:1000,1], z=t[:1000,2],marker=dict(size=1),
    line=dict(color='darkblue',width=2)))
fig.show()

plt.plot(pca.components_[0])
plt.show()

pca.components_.shape