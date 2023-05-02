import torch 
import torch.multiprocessing as mp
from Robot_5link_Env import RobotEnv, env_replay
from Robot_5link import get_coords, S, a, l
from sparse_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time
from utils import act_preprocessing, stack_arrays
from torch.distributions import MultivariateNormal
from env_config import *
from optimized_functions import calc_jnt_err, PDControl
from optimized_functions_5L import create_store_state
from Networks import Actor

def run_episode(env: RobotEnv, actor:Actor):
    done = False
    state = env.get_state()
    noise = .1*tau_max
    t = 0
    coord_list = [state[0]]
    feat_list = [state[1]]
    r = np.linalg.norm(np.ones(5)*jnt_vel_max/5)
    score = 0
    store_state = create_store_state(env)
    use_PID=False
    pid_steps = 0
    actor_steps = 0
    while not done:
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
        x, jnt_err, jnt_dedt, w = act_preprocessing(state_, env.weights,single_value=True,device=actor.device)
        with torch.no_grad():
            action = actor.forward(x, jnt_err, jnt_dedt,w)
            action += torch.normal(torch.zeros_like(action).to(actor.device),noise)
        t = env.t_step
        new_state, reward, done, info = env.step(action,use_PID=use_PID)
        new_store_state = create_store_state(env)
        if use_PID:
            action = info['action']
        env.store_transition(store_state, action, reward, new_store_state, done,t)
        store_state = new_store_state
        state = new_state
        coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
        if np.linalg.norm(env.jnt_err) < r:
            pid_steps += 1
            use_PID = True
        else:
            actor_steps += 1
        t+=1
        score += reward

    print('finished, actor_steps',actor_steps,'PID_steps', pid_steps)
    return env, score

actor = Actor(name='temp')
actor.load(load_as='chckptn_supervised_actor')
env = RobotEnv()
