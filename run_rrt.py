from Robot_5link_Env import RobotEnv, env_replay
from Robot_5link import get_coords, S, a, l
from spare_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from env_config import *
from rrt_5link import RRT_star
from trajectory import Trajectory_v2
from optimized_functions import calc_jnt_err, PDControl
from env_config import thres as r_thres

def run_rrt(env:RobotEnv):
    t_list = []
    k = 0
    max_iter = 1
    while len(t_list)==0 and k < max_iter:
        start = tuple(env.start)
        goal = tuple(env.goal)
        steps = 25
        thres = np.sqrt(5*.003)**2 * (steps/25)
        n = 15
        r = np.linalg.norm(np.ones(5)*jnt_vel_max/5)
        d = np.linalg.norm(jnt_vel_max*dt*steps*np.ones(5))
        max_samples = 4000
        rrt = RRT_star(start,goal,max_samples,r,d,thres,n,steps,env)
        t_list,traj = rrt.rrt_search()
        # obs = rrt.obs_dict
        k+=1

    if len(t_list) > 0:
        env.t_step = 0
        env.th = env.start
        env.w = np.zeros(env.th.shape, dtype=float)
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
        state = env.get_state()
        score = 0
        done = False
        for i in range(tau_list.shape[0]):
            tau = tau_list[i,:]
            t = env.t_step
            nxt_state, reward, done, info = env.step(tau,eval=True)
            env.store_transition(state, tau, reward, nxt_state, done, t)
            state = nxt_state
            score += reward
            th, w = env.th, env.w
        print("RRT steps", tau_list.shape[0])
        if done:
            print("ended with RRT")
        else:
            print("ending with PD control")
        PD_steps = 0
        while not done:
            tau = np.zeros(5, dtype=float)
            t = env.t_step
            nxt_state, reward,done, info = env.step(tau, use_PID=True,eval=True)
            action = info['action']
            env.store_transition(state, action, reward, nxt_state,done, t)
            score += reward
            PD_steps += 1
        print("steps with PD:", PD_steps)
    else:
        score = 0
        converged = False
    # if not done:
    #     while not done:

    return env, score, converged