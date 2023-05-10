import warnings
warnings.filterwarnings('ignore')
import numpy as np
from env_config import *
from optimized_functions_5L import *
from Object_v2 import rand_object
from Robot_5link_Env import RobotEnv

for i in range(100):
    env = RobotEnv(num_obj=3)
    th = env.th
    proxs = []
    for o in env.objs:
        obj_pos = o.curr_pos
        prox = proximity(obj_pos,th, a,l,S)
        proxs.append(prox)
    print(i,np.round(proxs,2))