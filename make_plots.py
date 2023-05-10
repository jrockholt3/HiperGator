import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d 
from env_config import *
from sparse_tnsr_replay_buffer import ReplayBuffer
import numpy as np
import pickle

def plot_scene(coords,feats):
    xx = coords[:,1]
    yy = coords[:,2]
    zz = coords[:,3]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xx[feats==rob_label],yy[feats==rob_label],zz[feats==rob_label],'b')
    ax.scatter3D(xx[feats==obj_label],yy[feats==obj_label],zz[feats==obj_label],'r')
    r = np.sum(S[1:])/res + np.sum(l)/res
    ax.set_xlim3d(left=0, right=2*r)
    ax.set_ylim3d(bottom=0, top=2*r)
    ax.set_zlim3d(bottom=0, top=r+S[0]/res)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_loss(train_loss, val_loss):
    fig = plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.show()

# buffer = ReplayBuffer(file='train_data')
# buffer = buffer.load()

# state, w, a, reward, nxt_state, done = buffer.sample_buffer(batch_size=1)
# coords = nxt_state[0][0]
# feats = nxt_state[1][0]
# feats = feats[:,0]
# # print(feats.shape)
# # print(coords[0])
# # print(feats)
# plot_scene(coords,feats)

file = open('loss_hist_0509.pkl','rb')
train_loss = pickle.load(file)
file = open('val_hist_0509.pkl','rb')
val_loss = pickle.load(file)
plot_loss(train_loss, val_loss)
print(train_loss[-1])