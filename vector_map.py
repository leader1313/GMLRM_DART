from tools.Data.Load import Load
from tools.Data.Save import Save
from tools.Learning.GMLRM import GMLRM
import sys,subprocess
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.externals import joblib

save = Save('data/')
load = Load('data/')

command = ("ls data/action | grep")
Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
Num_goal = 2

filename = 'GMLRM_model/learner.pickle'
model = joblib.load(filename)

L = 10
l = L*1j

Y, X = np.mgrid[-0.42:0.24:l,-1.1:-0.35:l]
state = np.zeros((6,L,L))
state[0,:] = -0.7751
state[1,:] = -0.7580
state[2,:] = -0.2769
state[3,:] = 0.0494
state[4,:] = X
state[5,:] = Y
U0 = np.zeros((L,L))
V0 = np.zeros((L,L))
U1 = np.zeros((L,L))
V1 = np.zeros((L,L))


# state = load.num_to_ten(state).float()
print(state[:,1,1].shape)
for i in range(L):
    for j in range(L):
        a = model.predict(state[:,i,j])
        U0[i,j] = a[0][0]
        V0[i,j] = a[0][1]
        
        U1[i,j] = a[1][0]
        V1[i,j] = a[1][1]

w0 = -1.1
w1 = -0.35
w2 = -0.42
w3 = 0.24

fig = plt.figure(figsize=(9, 5))
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1,1])

ax0 = fig.add_subplot(gs[0, 0])
strm = ax0.streamplot(X, Y, U0, V0, color=U0, linewidth=2, cmap='autumn')
fig.colorbar(strm.lines)
ax0.set_title('Mixture 1')

ax1 = fig.add_subplot(gs[0, 1])
strm = ax1.streamplot(X, Y, U1, V1, color=U1, linewidth=2, cmap='autumn')
fig.colorbar(strm.lines)
ax1.set_title('Mixture 2')

seed_points = np.array([[-0.75006, -0.75796, -0.40994], [-0.30187, 0.04942, -0.10977]])

ax2 = fig.add_subplot(gs[1, 0])
strm = ax2.streamplot(X, Y, U0, V0, color=U0, linewidth=2,
                     cmap='autumn', start_points=seed_points.T)
fig.colorbar(strm.lines)

ax2.plot(seed_points[0], seed_points[1], 'bo')
ax2.set(xlim=(w0, w1), ylim=(w2, w3))

seed_points = np.array([[-0.75006, -0.75796, -0.40994], [-0.30187, 0.04942, -0.10977]])

ax3 = fig.add_subplot(gs[1, 1])
strm = ax3.streamplot(X, Y, U1, V1, color=U1, linewidth=2,
                     cmap='autumn', start_points=seed_points.T)
fig.colorbar(strm.lines)

ax3.plot(seed_points[0], seed_points[1], 'bo')
ax3.set(xlim=(w0, w1), ylim=(w2, w3))

plt.tight_layout()
plt.show()