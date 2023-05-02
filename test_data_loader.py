import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pipe, Process, Queue
import pickle
from sparse_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor
from cpu_worker import supervised_data_loader
from time import time, sleep
from central_network import central_network
import numpy as np

train_buffer = ReplayBuffer(max_size=int(1e6), jnt_d=5, time_d=6,file='train_data')
train_buffer = train_buffer.load()
eval_buffer = ReplayBuffer(max_size=int(1e6), jnt_d=5, time_d=6,file='rrt_data8')
eval_buffer = eval_buffer.load()
batch_size = 512

q = Queue(maxsize=15)
q_val= Queue(maxsize=15)

def data_loader(buffer,eval_buffer,shutdown):
    run = True 
    wait_time = .1
    max_idle_time = 10
    max_iter = int(np.ceil(max_idle_time/wait_time))
    i = 0
    while run:
        if shutdown.poll():
            print('shutting down')
            run = False
            stop = shutdown.recv()
        elif not q.full():
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
            if i%100==0: print('idle',i)
            i+=1
            if i > int(max_iter): # stop runing after 10 seconds 
                run = False
                print('exiting')
            sleep(wait_time)
# def data_loader(buffer):
#         states, actions, weights, _,_,_ = buffer.sample_buffer(batch_size)
#         q.put([states,actions,weights])

    # return states, actions, weights

# t = time()
# items = [train_buffer for _ in range(4)]
# pool = Pool(processes=4)
# pool.apply_async(data_loader,items)
# print('time', time() - t)
# print('len results', len(results))
# for thing in results:
#     print(thing)

# q = Queue(maxsize=15)
# q_val= Queue(maxsize=15)
send_shutdown, recv_shutdown = Pipe()

p1 = Process(target=data_loader, args=(train_buffer,eval_buffer,recv_shutdown))
p2 = Process(target=data_loader,args=(train_buffer,eval_buffer,recv_shutdown))
p3 = Process(target=data_loader,args=(train_buffer,eval_buffer,recv_shutdown))
p4 = Process(target=data_loader,args=(train_buffer,eval_buffer,recv_shutdown))
p5 = Process(target=data_loader,args=(train_buffer,eval_buffer,recv_shutdown))
p6 = Process(target=data_loader,args=(train_buffer,eval_buffer,recv_shutdown))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
while not q_val.full():
    # print('waitng on q_val', q_val.full())
    sleep(.1)
i = 0
t2 = time()
while not q.empty() and i < 100:
    stuff = q.get()
    i+=1
    print(len(stuff))
    sleep(.01)
t3 = time()
print('que went dry in', t3-t2)
for i in range(1000):
    send_shutdown.send(True)
    sleep(.001)
while not q.empty():
    try:
        stuff = q.get(timeout=.1)
    except:
        print('empty')
    print(len(stuff))
while not q_val.empty():
    try:
        stuff = q_val.get(timeout=.1)
    except:
        print('empyt')
    print(len(stuff))
print('trying to join things')
send_shutdown.close()
p1.kill()
p2.kill()
p3.kill()
p4.kill()
p5.kill()
p6.kill()

