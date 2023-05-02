import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
from Networks import Actor
from Agent import Agent 

agent = Agent(transfer=True,actor_name="actor_0501",critic_name="critic_0501")
actor = Actor(name='cpu_worker',device='cpu')
actor.load_state_dict(agent.actor.state_dict())
# env, score = run_episode(env, actor)
# replay_from_memory(env)