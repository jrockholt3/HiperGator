import numpy as np
from numpy import pi as pi
from Robot_5link_Env import RobotEnv
import Robot_5link
from Robot_5link import S, l, a
import time as T
from time import time
from Object_v2 import Cylinder
from run_rrt import run_rrt
from path_replay_5L_from_memory import replay_from_memory
# from pathos.threading import ThreadPool
# from Networks import Actor 
# from pathos.parallel import ParallelPool
# from pathos.multiprocessing import ProcessPool
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool
from env_config import *
# phi = np.pi/4
# p = np.array([.3,.15*np.cos(phi),.15*np.sin(phi)+.3])
# u = np.array([0, np.cos(phi), np.sin(phi)])
# th_arr = Robot_5link.reverse(p,u,S,a,l)
# th = th_arr[0,:]
# pi = np.pi
# th = pi/4
# start = np.zeros(5, dtype=float)
# goal = start.copy()
# goal[1] = np.pi/4
# env = RobotEnv(start=start, goal=goal)
# actor = Actor(1,5,4,'temp',device='cpu')
# env = RobotEnv(num_obj=3)

# num_workers = 5
# workers = [env for _ in range(num_workers)]
# pool = Pool()
# results = pool.map(run_rrt, workers)
# best_score = np.inf*-1
# for tup in results:
#     if tup[2]:
#         if tup[1] > best_score:
#             env = tup[0]
#             best_score = tup[1]
# pool = ParallelPool(nodes=3)
# pool = ProcessPool(node=10)

goal = np.ones(5)*np.pi/4
start = np.zeros(5,dtype=float)
env = RobotEnv(num_obj=1)#,start=start, goal=goal)
env,score,converged = run_rrt(env)
print(score)
replay_from_memory(env)

t_steps = env.memory.time_step
actions = env.memory.action_memory
errs = env.memory.jnt_err_memory


actions = actions[t_steps<np.inf]
# pos = pos[t_steps<np.inf]
# dedt = dedt[t_steps<np.inf]
errs = errs[t_steps<np.inf]
t_steps = t_steps[t_steps<np.inf]


fig = plt.figure()
plt.plot(np.arange(actions.shape[0]), actions)
fig2 = plt.figure()
plt.plot(np.arange(errs.shape[0]), errs)
plt.show()

file = open('env.pkl','wb')
pickle.dump(env,file)
file = open('action_mem.pkl','wb')
pickle.dump(actions, file)
file = open('time_mem.pkl','wb')
pickle.dump(t_steps, file)

# t = time()
# env,score,converged = run_rrt(env)
# if converged:
#     replay_from_memory(env)

# workers = [env, env, env]
# env.reset()
# worker = [env]
# pool = ThreadPool()
# t = time()
# for i in range(n):
#     output = pool.map(run_rrt,worker)
#     env.reset()
# print('with pooling', (time()-t)/n)

# env.reset()
# env,score = run_episode(env)
# print('PD score',score)
# replay_from_memory(env)


# print(coords[0,:], feats.shape[0])

# cyl = Cylinder(r=.05)
# cyl.plot_cloud()
# print(cyl.original.shape)
# T = T_arr[1,:,:]
# Rot_y = np.array([[np.cos(pi/2), 0, np.sin(pi/2), 0],
#                 [0, 1, 0, 0],
#                 [-np.sin(pi/2), 0, np.cos(pi/2), 0],
#                 [0,0,0,1]])
# T=T@Rot_y
# print(Rot_y)
# print(T.shape)
# cyl.transform(T)
# cyl.plot_cloud()

# Rot_x = np.array([[1,0,0,0],
#                   [0, np.cos(pi/2), -np.sin(pi/2), 0],
#                   [0, np.sin(pi/2), np.cos(pi/2), 0],
#                   [0,0,0,1]])
# T = T_arr[2,:,:] 
# T = T@Rot_x
# cyl.transform(T)
# cyl.plot_cloud()

# T = T_arr[4,:,:]
# T = T@Rot_y
# cyl.transform(T)
# cyl.plot_cloud()