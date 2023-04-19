import numpy as np
import torch
from Agent import Agent
from Robot_5link_Env import RobotEnv
from env_config import tau_max
from optimized_functions import calc_jnt_err
from utils import stack_arrays, create_stack
from time import time 
import pickle
from pathos.threading import ThreadPool
from pathos.parallel import ParallelPool
from Parallel_Robot_Env import run_episode, run_rrt
from Networks import Actor
import logging
logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)


score_hist_file = 'score_hist_0308'
loss_hist_file = 'loss_hist_0308'
act_name = 'actor_0413'
crit_name = 'critic_0413'
alpha = .001 # actor lr
beta = .002 # critic lr
gamma = .99
top_only = False
transfer = False
load_check_ptn = False
load_memory = False
has_objs = True
num_obj = 4
top_only = False
epochs = 5 # number of epochs to train over the collected data per episode
num_workers = 5
episodes = 10 
best_score = -np.inf 
n = 30 # number of episodes to calculate the average score
batch_size = 128 # batch size

agent = Agent(alpha=alpha,beta=beta,batch_size=batch_size,max_size=int(1e6),
                noise=.01*tau_max,e=.1,enoise=.01*tau_max,
                actor_name=act_name, critic_name=crit_name)
rng = np.random.default_rng()

if load_check_ptn:
    agent.load_models()
    file = open('tmp/' + score_hist_file + '.pkl', 'rb')
    loss_hist = pickle.load(file)
    file = open('tmp/' + loss_hist_file + '.pkl', 'rb')
    score_history = pickle.load( file)
else:
    score_history = []
    loss_hist = []

if load_memory:
    agent.load_memory()

saved_checkpoint = False

actor = Actor(1,jnt_dim=5,D=4,name='cpu_copy',device='cpu')
thred_pool = ThreadPool()
parallel_pool = ParallelPool(nodes=4)
for i in range(episodes):
    t1 = time()
    actor.load_state_dict(agent.actor.state_dict())
    workers = [RobotEnv(actor,agent.noise) for i in range(num_workers-1)]

    output = pool.map(run_episode,workers)
    env, score, converged = run_rrt(RobotEnv(actor,agent.noise))

    mems,scores=[],[]
    if converged:
        mems.append(env.memory)
    for tup in output:
        env,score = tup[0], tup[1]
        mems.append(env.memory)
        scores.append(score)
    score = np.mean(score)

    agent.memory.add_data(mems)

    mem_num = min(agent.memory.mem_cntr,agent.memory.mem_size)
    if mem_num < batch_size:
        batch_size_ = mem_num
        n_batch = 1
    else: 
        n_batch = int(np.floor(mem_num/batch_size))
        batch_size_ = batch_size
    batch = rng.choice(mem_num,size=n_batch*batch_size_,replace=False)

    loss = 0
    for _ in range(epochs):
        for j in range(n_batch):
            loss+=agent.learn(use_batch=True,batch=batch[j*batch_size:(j+1)*batch_size])

    agent.memory.clear()
        
    score_history.append(score)
    loss_hist.append(loss/(epochs*n_batch))

    if np.mean(score_history[-n:]) > best_score and i > n:
    # if i%10 == 0:
        saved_checkpoint = True
        agent.save_models()
        best_score = np.mean(score_history[-n:])
        file = open('tmp/' + score_hist_file + '.pkl', 'wb')
        pickle.dump(loss_hist,file)
        file = open('tmp/' + loss_hist_file + '.pkl', 'wb')
        pickle.dump(score_history, file)

    t2 = time()
    print('episode', i, 'train_avg %.2f' %np.mean(score_history[-n:]) \
        ,'final jnt_err', np.round(env.jnt_err,2),'time %.2f' %(t2-t1), 'avg loss', loss/n_batch)


best_score = np.mean(score_history[-n:])
file = open('tmp/' + score_hist_file + '.pkl', 'wb')
pickle.dump(loss_hist,file)
file = open('tmp/' + loss_hist_file + '.pkl', 'wb')
pickle.dump(score_history, file)

if not saved_checkpoint:
    agent.save_models()
