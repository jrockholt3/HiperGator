from Robot_5link_Env import RobotEnv
from spare_tnsr_replay_buffer import ReplayBuffer
from Parallel_Robot_Env import run_rrt
from Networks import Actor


actor = Actor(1,5,D=4,name='cpu_copy')
memory = ReplayBuffer(int(1e6), jnt_d=5, time_d=6, file='rrt_data')

k = 0
loop_max = 1.5e6
while memory.mem_cntr < memory.mem_size and k < loop_max:
    env = RobotEnv(actor=actor,noise=.1*30)
    env,score,converged = run_rrt(env)
    if converged:
        memory.add_data([env.memory])
        memory.save()
    