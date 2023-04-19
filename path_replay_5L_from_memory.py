import platform
import matplotlib
# matplotlib.use('nbAgg'
# print(platform.system())
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
# from Robot_Env import dt, RobotEnv, jnt_vel_max
from env_config import dt, jnt_vel_max, damping as Z
from Robot_5link_Env import RobotEnv
from Robot_5link_Env import env_replay
from Object_v2 import rand_object
import Robot_5link
from Robot_5link import S, a, l 
from env_config import *
from env_config import *
from utils import stack_arrays
from rrt_5link import RRT_star
from support_classes import vertex
from trajectory import Trajectory
from optimized_functions import calc_jnt_err
from spare_tnsr_replay_buffer import ReplayBuffer

lims = Robot_5link.limits

show_box = False
use_PID = False
num_obj = 2
evaluation = True

def replay_from_memory(env:RobotEnv):

    # init figure
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    x_arr = []
    y_arr = []
    z_arr = []
    # obj1's data
    x_arr2,y_arr2,z_arr2 = [],[],[]
    t = 0
    score = 0
    actions = env.memory.action_memory
    dones = env.memory.terminal_memory
    i = 0
    env.th = env.start
    env.t_step = 0
    th = env.start
    done = False
    while not done and not dones[i]:
        action = actions[i,:]
        # print('action', action)
        _,_,done,_ = env.step(action,eval=True)
        th = env.th
        temp = Robot_5link.points(th,S,a,l)
        temp2 = Robot_5link.points(env.goal,S,a,l)
        temp = np.hstack((temp, temp2))
        x_arr.append(temp[0,:])
        y_arr.append(temp[1,:])
        z_arr.append(temp[2,:])
        temp = []
        for o in env.objs:
            temp.append(o.path(env.t_step))
        temp = np.vstack(temp)
        x_arr2.append(temp[:,0])
        y_arr2.append(temp[:,1])
        z_arr2.append(temp[:,2])
        i += 1

    x_arr = np.vstack(x_arr)
    y_arr = np.vstack(y_arr)
    z_arr = np.vstack(z_arr)
    x_arr2 = np.vstack(x_arr2)
    y_arr2 = np.vstack(y_arr2)
    z_arr2 = np.vstack(z_arr2)


    line, = ax.plot([],[],[], 'bo-', lw=2) # robot at t
    line2, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-1
    line3, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-2 
    line4, = ax.plot([],[],[], 'ro', lw=10) # obj at t
    line5, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-1
    line6, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-2
    # line7, = ax.plot([],[],[], 'k-', alpha=.3) # box

    j = int(np.round(x_arr.shape[0]))
    def update(i):
        global j
        # set robot lines
        thisx = x_arr[i,0:5]
        thisy = y_arr[i,0:5]
        thisz = z_arr[i,0:5]
        line.set_data_3d(thisx,thisy,thisz)

        thisx = x_arr[i,5:10]
        thisy = y_arr[i,5:10]
        thisz = z_arr[i,5:10]
        line2.set_data_3d(thisx,thisy,thisz)
        # n = 3
        # if i > n-1:
        #     lastx = [x_arr[i-n,0],x_arr[i-n,1],x_arr[i-n,2],x_arr[i-n,3]]
        #     lasty = [y_arr[i-n,0],y_arr[i-n,1],y_arr[i-n,2],y_arr[i-n,3]]
        #     lastz = [z_arr[i-n,0],z_arr[i-n,1],z_arr[i-n,2],z_arr[i-n,3]]
        #     line2.set_data_3d(lastx,lasty,lastz)
        # else:
        #     line2.set_data_3d(thisx,thisy,thisz)

        n = 0
        if i > n-1:
            lastx = x_arr[0,0:5]
            lasty = y_arr[0,0:5]
            lastz = z_arr[0,0:5]
            line3.set_data_3d(lastx,lasty,lastz)
        else:
            line3.set_data_3d(thisx,thisy,thisz)

        # set object lines 
        objx,objy,objz = x_arr2[i,:],y_arr2[i,:],z_arr2[i,:]
        line4.set_data_3d(objx,objy,objz)
        n = 3
        if i > n-1:
            lastx = x_arr2[i-n]
            lasty = y_arr2[i-n]
            lastz = z_arr2[i-n]
            line5.set_data_3d(lastx,lasty,lastz)
        else:
            line5.set_data_3d(objx,objy,objz)

        n = 6
        if i > n-1:
            lastx = x_arr2[i-n]
            lasty = y_arr2[i-n]
            lastz = z_arr2[i-n]
            line6.set_data_3d(lastx,lasty,lastz)
        else:
            line6.set_data_3d(objx,objy,objz)

        return line, line2, line3, line4, line5, line6

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    # lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d(lims[0,:])
    ax.set_xlabel('X')

    ax.set_ylim3d(lims[1,:])
    ax.set_ylabel('Y')

    ax.set_zlim3d(lims[2,:])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    N = x_arr.shape[0]
    speed = dt*1000

    ani = animation.FuncAnimation(
        fig, update, N, interval=speed, blit=False)

    # ani.save('file.gif')

    plt.show()
