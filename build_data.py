import warnings
warnings.filterwarnings("ignore")
from Robot_5link_Env import RobotEnv
from sparse_tnsr_replay_buffer import ReplayBuffer
from run_rrt import run_rrt
import numpy as np
import sys 

rng = np.random.default_rng()

def build_data(name):
    memory = ReplayBuffer(file=name)
    episode_num = 100
    k = 0
    loops = 0
    loop_max = 1000
    while memory.mem_cntr < memory.mem_size and k < episode_num and loops<loop_max:
        loops +=1
        num_obj = 3 #rng.choice([2,4,4,4,4,5])
        env = RobotEnv(num_obj=num_obj, use_goal=True)
        env,score,converged = run_rrt(env)
        if converged:
            k+=1
            memory.add_data([env.memory])
            memory.save()
            print(name,"converged, adding memory", memory.mem_cntr, k)

    print(name,'finished, mem_cntr',memory.mem_cntr)

if __name__ == "__main__":
    args = sys.argv
    name = args[1]
    build_data(name)
