import torch 
import torch.multiprocessing as mp
# from Robot_Env import RobotEnv
from Robot_5link_Env import RobotEnv, env_replay
from Robot_5link import get_coords, S, a, l
from spare_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time
# from pathos.threading import ThreadPool
# from PPO_Networks import Actor
# from Reduced_Networks import Actor
from utils import act_preprocessing, stack_arrays
from torch.distributions import MultivariateNormal
# from Robot_Env import tau_max, t_limit, dt
from env_config import *
from rrt_5link import RRT_star
from trajectory import Trajectory, Trajectory_v2
from optimized_functions import calc_jnt_err, PDControl
from env_config import thres as r_thres
# from Networks import Actor

class ParallelRobotEnv():
    def __init__(self, actor_state_dict, noise=.01*tau_max, use_PID=False):
        # super(ParallelRobotEnv,self).__init__()
        env = RobotEnv(num_obj=3)
        mem_size = int(np.round(2*t_limit/dt))
        env,state = env.reset()
        self.env = env
        # self.memory = ReplayBuffer(mem_size,jnt_d=3, time_d=6)
        # self.goal_memory = ReplayBuffer(mem_size, jnt_d=3, time_d=6)
        # self.actor = Actor(in_feat=1, jnt_dim=3,D=4,name='actor',device='cpu')
        # self.actor.load_state_dict(actor_state_dict)
        # self.noise = noise
        self.use_PID = use_PID
        
    def step(self, action, use_PID=False):
        return self.env.step(action, use_PID=use_PID)
    
    def choose_action(self, x, jnt_pos, jnt_goal):
        if not self.use_PID:
            with torch.no_grad():
                action = self.actor.forward(x, jnt_pos, jnt_goal)
            action += torch.normal(torch.zeros_like(action),self.noise)
        else: 
            action = torch.zeros(3, device='cuda')
        return action

    def reset(self):
        env, state = self.env.reset()
        self.env = env
        return self, state
        
    def get(self):
        return self.memory

def run_episode(env: RobotEnv):
    done = False
    state = env.get_state()
    t = 0
    coord_list = [state[0]]
    feat_list = [state[1]]
    score = 0
    while not done:
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
        x, jnt_pos, jnt_goal = act_preprocessing(state_,single_value=True,device=env.actor.device)
        with torch.no_grad():
            action = env.choose_action(x, jnt_pos, jnt_goal)
        t = env.t_step
        new_state, reward, done, info = env.step(action)
        env.memory.store_transition(state, action, reward, new_state, done,t)
        state = new_state
        coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
        t+=1
        score += reward

    print('finished')
    return env, score

def run_rrt(env:RobotEnv):
    t_list = []
    k = 0
    max_iter = 1
    while len(t_list)==0 and k < max_iter:
        start = tuple(env.start)
        goal = tuple(env.goal)
        steps = 15
        thres = np.sqrt(5*.003)**2 * (steps/25)
        n = 25
        r = np.linalg.norm(np.ones(5)*r_thres)
        d = np.linalg.norm(jnt_vel_max*dt*steps*1.5*np.ones(5))
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
            # obs_arr = obs[t]    
            # jnt_err = calc_jnt_err(th,np.array(targ))
            # dedt = -1*w
            # tau = PDControl(jnt_err, dedt)
            tau = tau_list[i,:]
            # nxt_th, nxt_w, reward, t, flag = env_replay(th,w,t,np.array(targ),obs,1,S,a,l)
            t = env.t_step
            nxt_state, reward, done, info = env.step(tau,eval=True)
            # if np.all(nxt_th <= thres) and np.all(nxt_w <= thres):
            #     done = True
            # elif t >= t_limit/dt:
            #     done = True
            # else:
            #     done = False
            # obs_arr = obs[t]   
            # rob_coords,rob_feats = rrt_get_coords(t,nxt_th,obs_arr) 
            # nxt_state = (coords,feats, nxt_th, nxt_w)
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


# workers = [ParallelRobotEnv() for i in range(6)]
# t1 = time()
# pool = ThreadPool()
# # pool = ParallelPool()
# # pool = multiprocessing.Pool(len(workers))
# # pool = ProcessPool()

# mems = pool.map(run_episode,workers)
# t2 = time()
# print('time', t2-t1)

# print(mems)