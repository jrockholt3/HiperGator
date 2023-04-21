import torch 
import torch.multiprocessing as mp
from Robot_5link_Env import RobotEnv, env_replay
from Robot_5link import get_coords, S, a, l
from spare_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time
from utils import act_preprocessing, stack_arrays
from torch.distributions import MultivariateNormal
from env_config import *
from optimized_functions import calc_jnt_err, PDControl
from Networks import Actor


def run_episode(env: RobotEnv, actor:Actor):
    done = False
    state = env.get_state()
    noise = .1*tau_max
    t = 0
    coord_list = [state[0]]
    feat_list = [state[1]]
    score = 0
    while not done:
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
        x, jnt_pos, jnt_goal = act_preprocessing(state_,single_value=True,device=actor.device)
        with torch.no_grad():
            action = actor.forward(x, jnt_pos, jnt_goal)
            action += torch.normal(torch.zeros_like(action).to(actor.device),noise)
        t = env.t_step
        new_state, reward, done, info = env.step(action)
        env.memory.store_transition(state, action, reward, new_state, done,t)
        state = new_state
        coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
        t+=1
        score += reward

    print('finished')
    return env, score
