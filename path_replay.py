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
from Object_v2 import rand_object
from utils import stack_arrays
from search_space import SearchSpace
from support_classes import vertex
from trajectory import Trajectory
from optimized_functions import calc_jnt_err

show_box = False
use_PID = False
num_obj = 2
evaluation = True
actor_name = 'actor_0308'
critic_name = 'critic_0308'

class Box():
    def __init__(self):
        a = .08
        self.x_arr = np.array([-a,a,a,-a,-a,-a,a,a,a,a,a,a,-a,-a,-a,-a])
        self.y_arr = np.array([a,a,-a,-a,a,a,a,a,a,-a,-a,-a,-a,-a,-a,a])
        self.z_arr = np.array([a,a,a,a,a,-a,-a,a,-a,-a,a,-a,-a,a,-a,-a])
        self.pos = np.array([0,0,0])

    def render(self,pos):
        return self.x_arr+pos[0], self.y_arr+pos[1], self.z_arr+pos[2]

def gen_centers(x,y,z):
    centers = []
    n = 5
    vec = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    centers.append(np.array([x[0],y[0],z[0]]))
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[0])

    vec = np.array([x[2]-x[1], y[2]-y[1], z[2]-z[1]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[n])
    
    return np.vstack(centers)


# init figure
fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')


# Generate Path
path = []
traj = []
max_iter = 5
i = 0
while len(path)==0 and i < max_iter:
    env = RobotEnv2(num_obj=3)
    X = SearchSpace((750, np.pi, .9*np.pi, .9*np.pi), env)
    start = tuple(env.start)
    goal = tuple(env.goal)
    global_goal = np.array(goal)
    print('goal', goal)
    steps = 5
    thres = np.linalg.norm([.003,.003,.003])*(steps/25)
    n = 25
    r = np.linalg.norm(jnt_vel_max*dt*steps*np.ones(3)/3)
    d = np.linalg.norm(jnt_vel_max*dt*steps*1.5*np.ones(3))
    max_samples = int(3000)
    rrt = RRT_star(X, start, goal, max_samples, r, d, thres,n=n,steps=steps)
    path,traj = rrt.rrt_search()
    obs = rrt.get_obs()

    i += 1

# rrt.plot_graph(every=1)

x_arr = []
y_arr = []
z_arr = []
# obj1's data
x_arr2,y_arr2,z_arr2 = [],[],[]
robot = robot_3link()
a = robot.aph
l = robot.links
S = robot.S
th = np.array(start)
w = np.zeros_like(th)
t = 0
score = 0
tau_list = Trajectory(traj)
for targ in tau_list:
    obs_arr = obs[t]
    # tau_in = tau + Z*w
    t = int(t)
    nxt_th, nxt_w, reward, t, flag = env_replay(th, w, t, np.array(targ), obs, 1,S,a,l)
    # nxt_th = tau * dt**2/2 + w*dt + th
    # nxt_w = tau*dt + w
    # t+=1

    robot.set_pose(th)
    temp = robot.forward()
    temp2 = robot.forward(th=global_goal)
    temp = np.hstack((temp, temp2))
    x_arr.append(temp[0,:])
    y_arr.append(temp[1,:])
    z_arr.append(temp[2,:])
    temp = []
    for i in range(obs_arr.shape[1]):
        temp.append(obs_arr[:,i])
    temp = np.vstack(temp)
    x_arr2.append(temp[:,0])
    y_arr2.append(temp[:,1])
    z_arr2.append(temp[:,2])

    th, w = nxt_th, nxt_w

print('score ', score) 

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
    thisx = [x_arr[i,0],x_arr[i,1],x_arr[i,2],x_arr[i,3]]
    thisy = [y_arr[i,0],y_arr[i,1],y_arr[i,2],y_arr[i,3]]
    thisz = [z_arr[i,0],z_arr[i,1],z_arr[i,2],z_arr[i,3]]

    line.set_data_3d(thisx,thisy,thisz)
    thisx = [x_arr[i,4],x_arr[i,5],x_arr[i,6],x_arr[i,7]]
    thisy = [y_arr[i,4],y_arr[i,5],y_arr[i,6],y_arr[i,7]]
    thisz = [z_arr[i,4],z_arr[i,5],z_arr[i,6],z_arr[i,7]]
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
        lastx = [x_arr[0,0],x_arr[0,1],x_arr[0,2],x_arr[0,3]]
        lasty = [y_arr[0,0],y_arr[0,1],y_arr[0,2],y_arr[0,3]]
        lastz = [z_arr[0,0],z_arr[0,1],z_arr[0,2],z_arr[0,3]]
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
if show_box:
    N = x_box[0,:].shape[0]
    speed = dt*10000/2
else:
    N = x_arr.shape[0]
    speed = dt*1000

ani = animation.FuncAnimation(
    fig, update, N, interval=speed, blit=False)

# ani.save('file.gif')

plt.show()

rrt.plot_graph(every=1, add_path=True, path=path)