import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
# from Networks import Actor
# from Agent import Agent 
from Robot_5link_Env import RobotEnv
# agent = Agent(transfer=True,actor_name="actor_0501",critic_name="critic_0501")
# actor = Actor(name='cpu_worker',device='cpu')
# actor.load_state_dict(agent.actor.state_dict())
# # env, score = run_episode(env, actor)
# # replay_from_memory(env)
from Object_v2 import rand_object
for i in range(10):
    # q = np.random.choice([1,2,3,4])
    # o = rand_object(q=int(q))
    # print('q',q,'start',np.round(o.start,2))
    env = RobotEnv()
    # for o in env.objs:
    #     print(np.round(o.curr_pos,2), np.round(o.start,2))
