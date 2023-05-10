import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from Networks import SupervisedActor
from utils import crit_preprocessing 
from sparse_tnsr_replay_buffer import ReplayBuffer
import pickle
from time import time, sleep
from multiprocessing import Pipe, Process, Queue

batch_size = 512
epochs = 50
num_workers = 6
q_size = 15

actor = SupervisedActor(name='supervised_actor_0509')
optimizer = Adam(params=actor.parameters())

train_buffer = ReplayBuffer(max_size=int(1e6), jnt_d=5, time_d=6,file='train_data')
train_buffer = train_buffer.load()
eval_buffer = ReplayBuffer(max_size=int(1e6), jnt_d=5, time_d=6,file='rrt_data10')
eval_buffer = eval_buffer.load()

q = Queue(maxsize=q_size)
q_val= Queue(maxsize=q_size)
def data_loader(buffer,eval_buffer):
    run = True 
    wait_time = .1
    max_idle_time = 600
    max_iter = int(np.ceil(max_idle_time/wait_time))
    i = 0
    while run:
        if not q.full():
            states, actions, weights, _,_,_ = buffer.sample_buffer(batch_size)
            try:
                q.put([states,actions,weights],timeout=1)
            except:
                sleep(.1)
            i = 0
        elif not q_val.full():
            states,actions,weights,_,_,_ = eval_buffer.sample_buffer(batch_size)
            try:
                q_val.put([states,actions,weights],timeout=1)
            except:
                sleep(.1)
        else:
            # if i%100==0: print('idle',i)
            i+=1
            if i > int(max_iter): # stop runing after 10 seconds 
                run = False
                print('exiting')
            sleep(wait_time)

def save_files(loss_hist,val_hist):
    loss_file = "loss_hist_0508.pkl"
    val_file = "val_hist_0508.pkl"
    file = open(loss_file,'wb')
    pickle.dump(loss_hist,file)
    file = open(val_file,'wb')
    pickle.dump(val_hist,file)

#admin
loss_hist = []
val_hist = []
best_loss = np.inf
n_batch = int(np.floor(train_buffer.mem_cntr/batch_size))
val_batch = int(np.floor(eval_buffer.mem_cntr/batch_size))
k = 0
keep_going = True 
early_stop = 0

# setting up processes
print('starting processes')
processes = [Process(target=data_loader,args=(train_buffer,eval_buffer)) for _ in range(num_workers)]
for p in processes:
    p.start()
print('filling up q')
while not q.full():
    sleep(1)
print('filling up q_val')
while not q_val.full():
    sleep(1)
 
try: 
    while k < epochs and keep_going:
        running_loss = 0
        print('...learning...', k)
        for i in range(n_batch):
            temp_loop = 0
            # t = time()
            while q.empty():
                # print('recovering')
                sleep(5)
            stuff = q.get()
            states,actions,weights = stuff[0],stuff[1],stuff[2]
            # print("loading time", time()-t)
            # t = time()
            x,jnt_err,jnt_dedt,w,a_ = crit_preprocessing(states,actions,weights)
            # print("preprocessing", time()-t)
            # t = time()
            optimizer.zero_grad()
            a = actor.forward(x,jnt_err,jnt_dedt,w)
            loss = F.mse_loss(a,a_)
            loss.backward()
            optimizer.step()
            # print("learning", time()-t)
            running_loss += loss.item()
        batch_loss = running_loss/n_batch
        loss_hist.append(batch_loss)

        running_loss = 0
        print('...eval...', k)
        for i in range(val_batch):
            with torch.no_grad():
                temp_loop = 0
                while q_val.empty():
                    # print('recovering')
                    sleep(5)
                stuff = q_val.get()
                states,actions,weights = stuff[0],stuff[1],stuff[2]
                x,jnt_err,jnt_dedt,w,a_ = crit_preprocessing(states,actions,weights)
                a = actor.forward(x,jnt_err,jnt_dedt,w)
                loss = F.mse_loss(a,a_)
                running_loss += loss.item()
        val_loss = running_loss/val_batch
        val_hist.append(val_loss)
	
        k+=1
        if val_loss <= best_loss:
            actor.save_checkpoint(save_as="chckptn_"+actor.name)
            save_files(loss_hist,val_hist)
        else: 
            early_stop += 1
            if early_stop > 20:
                keep_going = False

        print('epoch',k,'training loss', batch_loss, 'val loss', val_loss)
    actor.save_checkpoint()
    save_files(loss_hist,val_hist)

    for p in processes:
        print('killing')
        p.kill()
except Exception as e:
    print(e)
    for p in processes:
        print('killing')
        p.kill()

