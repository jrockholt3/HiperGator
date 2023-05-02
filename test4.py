import warnings
import torch 
warnings.filterwarnings("ignore")
# from sparse_tnsr_replay_buffer import ReplayBuffer
import numpy as np
import pickle 
from Robot_5link_Env import RobotEnv
from Agent import Agent 
from Networks import Actor 
from multiprocessing import Process, Pipe, Queue
from sparse_tnsr_replay_buffer import ReplayBuffer
from time import time,sleep
from utils import stack_arrays, act_preprocessing
from env_config import * 
from optimized_functions_5L import create_store_state

agent = Agent(transfer=True,actor_name="actor_0501",critic_name="critic_0501")
agent.memory = None 
batch_size = 128
num_workers = 2
num_rrt = 0
n_batch = 5
n = 100
q_size = 3
epochs = int(1)
rrt_guys = []
workers = []
data_q = Queue(maxsize=q_size)

def cpu_worker(recv_dict, send_score, init_dict):
    print('started worker')
    from Networks import Actor
    from sparse_tnsr_replay_buffer import ReplayBuffer
    from Robot_5link_Env import RobotEnv
    actor = Actor(name="cpu_worker",device='cpu')
    actor.load_state_dict(init_dict)
    for name,param in actor.named_parameters():
        param.requires_grad = False
    buffer = ReplayBuffer(max_size=int(1e4))
    noise = .1*tau_max
    radius = np.linalg.norm(np.ones(5)*jnt_vel_max/10)
    run = True 
    print('starting loop')
    while np.any(run):
        env = RobotEnv()
        done = False
        state = env.get_state()
        t = 0
        coord_list = [state[0]]
        feat_list = [state[1]]
        score = 0
        store_state = create_store_state(env)
        use_PID=False
        print("starting episode")
        while not np.any([done]):
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
            if np.linalg.norm(env.jnt_err) < radius:
                use_PID = True
            t+=1
            score += reward

            if not data_q.full() and buffer.mem_cntr>batch_size:
                print('putting in data')
                s, w, a, r, nxt_s, terms = buffer.sample_buffer(batch_size) 
                data_q.put((s,w,a,r,nxt_s,terms),timeout=1)
            if recv_dict.poll():
                print('recieved')
                state_dict = recv_dict.recv()
                actor.load_state_dict(state_dict)
                for name,param in actor.named_parameters():
                        param.requires_grad = False

        send_score.send(score)
        buffer.add_data([env.memory])

    # data loader for RRT data
def data_loader(buffer, data_q):
    run = True 
    wait_time = .1
    max_idle_time = 600
    max_iter = int(np.ceil(max_idle_time/wait_time))
    i = 0
    while run:
        if not data_q.full():
            s, a, w, r, nxt_s, dones = buffer.sample_buffer(batch_size)
            try:
                q.put([s,a,w,r,nxt_s,dones],timeout=1)
            except:
                sleep(.1)
            i = 0
        else:
            i+=1
            if i > int(max_iter): # stop runing after 10 seconds 
                run = False
                print('exiting')
            sleep(wait_time)

def save_files(score_hist,loss_hist):
    loss_file = "loss_hist_0501.pkl"
    score_file = "score_hist_0501.pkl"
    file = open(loss_file,'wb')
    pickle.dump(loss_hist,file)
    file = open(score_file,'wb')
    pickle.dump(score_hist,file)

def get_cpu_dict(state_dict):
    for k,v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict

# Starting up Processes 
init_dict = agent.actor.state_dict()
init_dict = get_cpu_dict(init_dict)
mail_list = []
recv_list = []
print('starting workerts')
recv_dict,send_dict = Pipe()
recv_score,send_score = Pipe()
cpu_worker(recv_dict,send_score,init_dict)
# for i in range(num_workers):
#     print('woker',i)
#     recv_dict,send_dict = Pipe()
#     recv_score,send_score = Pipe()
#     mail_list.append(send_dict)
#     recv_list.append(recv_score)
#     p = Process(target=cpu_worker,args=(recv_dict, send_score, init_dict))
#     workers.append(p)

for p in workers:
    p.start()
for p in workers:
    print(p.is_alive())

buffer = ReplayBuffer(file="train_data")
buffer = buffer.load()
rrt_guys = []
for i in range(num_rrt):
    print('starting rrt guys')
    p = Process(target=data_loader,args=([buffer,data_q]))
    rrt_guys.append(p)
    p.start()


# admin 
k = 0
score_hist = []
loss_hist = []
best_score = np.inf*-1
print('filling up queue')
while not data_q.full():
    sleep(1)
try: 
    print('starting to trian')
    while k < epochs and keep_going:
        running_loss = 0
        print('training')
        for i in range(n_batch):
            print(i)
            while q.empty():
                print('wating on data')
                sleep(1)
            b = data_q.get()
            running_loss += agent.learn(use_data=True,data=b)
        loss_hist.append(running_loss/n_batch)

        print('seding dict')
        for channel in mail_list:
            state_dict = agent.actor.state_dict()
            state_dict = get_cpu_dict(state_dict)
            channel.send(state_dict)
        score = 0
        n_score = 0
        print('asking for scores')
        for channel in recv_list:
            if channel.poll():
                score += channel.recv()
                n_scores += 1

        if n_scores > 0: 
            avg_score = np.mean(score_hist[-n:])
            score_hist.append(score/n_scores)
            if avg_score > best_score:
                print('check point')
                best_score = avg_score
                agent.save_models(chckptn=True)
        
        save_files(score_hist,loss_hist)
        print('epoch',k, 'average score', np.mean(score_hist[-n:]), 'critic_loss', loss_hist[-1])
        k+=1
    agent.save_models()
    save_files(loss_hist,val_hist)

    for p in workers:
        print('killing')
        p.kill()
    for p in rrt_guys:
        p.kill()
except Exception as e:
    print(e)
    for p in workers:
        print('killing')
        p.kill()
    for p in rrt_guys:
        p.kill()



