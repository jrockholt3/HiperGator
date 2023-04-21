import pickle
import matplotlib.pyplot as plt
import numpy as np
from run_episode import run_episode
from Networks import Actor
from Robot_5link_Env import RobotEnv
from path_replay_5L_from_memory import replay_from_memory

actor = Actor(1,5,4,'temp_actor',device='cpu')
env = RobotEnv()

env, score = run_episode(env, actor)
replay_from_memory(env)