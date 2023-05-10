import warnings
warnings.filterwarnings('ignore')
from Robot_5link_Env import RobotEnv, env_replay
from Robot_5link import get_coords, S, a, l
from sparse_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from env_config import *
from rrt_5link import RRT_star
from trajectory import Trajectory_v2
from optimized_functions import calc_jnt_err, PDControl
from optimized_functions_5L import create_store_state
from env_config import thres as r_thres
from path_replay_5L_from_memory import replay_from_memory
from time import time

def run_rrt(env:RobotEnv):
    t_list = []
    k = 0
    max_iter = 1
    while len(t_list)==0 and k < max_iter:
        start = tuple(env.start)
        goal = tuple(env.goal)
        steps = 15
        thres = np.sqrt(5*.03**2) #* (steps/25)
        n = 20
        r = np.linalg.norm(np.ones(5)*jnt_vel_max/10)
        d = np.linalg.norm(jnt_vel_max*dt*steps*np.ones(5))
        max_samples = 4000
        # print('starting rrt')
        rrt = RRT_star(start,goal,max_samples,r,d,thres,n,steps,env)
        t_list,traj = rrt.rrt_search()
        # rrt.plot_graph()
        # obs = rrt.obs_dict
        k+=1

    if len(t_list) > 0:
        env.t_step = 0
        env.th = env.start
        env.w = np.zeros(env.th.shape)
        converged = True
        t_arr = np.vstack(t_list)
        J_arr = np.vstack(traj)
        tau_list = Trajectory_v2(J_arr, t_arr)
        # tau_arr = Traject
        th = np.array(start)
        w = np.zeros_like(th,dtype=float)
        # obs_arr = obs[t]
        # coords, feats = rrt_get_coords(t,th,obs_arr)
        # jnt_err = calc_jnt_err(th, np.array(goal))
        # dedt = -1*w
        # state = env.get_state()
        score = 0
        done = False
        store_state = create_store_state(env)
        for i in range(tau_list.shape[0]):
            tau = tau_list[i,:]
            t = env.t_step
            nxt_state, reward, done, info = env.step(tau,eval=True)
            nxt_store_state = create_store_state(env)
            env.store_transition(store_state, tau, reward, nxt_store_state, done, t)
            store_state = nxt_store_state
            state = nxt_state
            score += reward
            th, w = env.th, env.w
            if done: 
                i = tau_list.shape[0] + 1
            # elif np.linalg.norm(env.jnt_err) < r:
                # i = tau_list.shape[0] + 1
        # print("RRT steps", tau_list.shape[0])
        # if done:
        #     print("ended with RRT")
        # else:
        #     print("ending with PD control")
        PD_steps = 0
        while not done:
            tau = np.zeros(5)
            t = env.t_step
            nxt_state, reward, done, info = env.step(tau,use_PID=True,eval=True)
            nxt_store_state = create_store_state(env)
            action = info['action']
            # converged = info['converged']
            env.store_transition(store_state, action, reward, nxt_store_state, done, t)
            store_state = nxt_store_state
            score += reward
            PD_steps += 1
        print("steps with RRT", len(tau_list), "steps with PD:", PD_steps, 'final jnt_err', env.jnt_err)
    else:
        score = 0
        converged = False
    # if not done:
    #     while not done:

    return env, score, converged


# env1 = RobotEnv(num_obj=4)#,start=np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]), goal=np.array([0.0,0.0,0.0,0.0,0.0]))
# env2 = env1.copy()
# env3 = env1.copy()
# print('limits', np.round(180*jnt_max/np.pi,2))
# print('goal1', np.round(180*env1.goal/np.pi,2), 'start1', np.round(180*env1.start/np.pi,2))
# # env1.weights = np.array([.99, .1, .1])
# t1 = time()
# env1, score1,converged1 = run_rrt(env1)
# print('run time is ', time()-t1)
# # env2.weights = np.array([.1, .99, .1])
# # env2,score2, converged2 = run_rrt(env2)
# # env3.weights = np.array([.1,.1,.99])
# # env3,score3,converged3 = run_rrt(env3)
# # print('score1', score1, converged1, 'score2', score2,converged2, 'score3', score3, converged3)
# print('converged', converged1)
# replay_from_memory(env1)
# # replay_from_memory(env2)
# # replay_from_memory(env3)