import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import MinkowskiEngine as ME
from sparse_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor as Actor, Critic
# from Reduced_Networks import Actor, Critic
import pickle
import gc 
from env_config import * 
from utils import act_preprocessing, crit_preprocessing

mse_loss = nn.MSELoss()

def check_memory():
    q = 0
    for obj in gc.get_objects():
        try:  
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                q += 1
        except:
            pass
    return q 

class Agent():
    def __init__(self, alpha=0.005,beta=0.01, gamma=.99, n_actions=5, 
                time_d=6, max_size=int(1e6), tau=0.1,
                batch_size=64,noise=.01*tau_max,e=.1,enoise=.1*tau_max,
                top_only=False,transfer=False,actor_name = 'actor',critic_name='critic',
                buff_name='replay_buffer'):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size,n_actions,time_d,file=buff_name)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.e = e
        self.enoise = enoise
        self.max_action = torch.tensor(torch.ones(n_actions)*tau_max, device='cuda')
        self.min_action = torch.tensor(torch.ones(n_actions)*-1*tau_max, device='cuda')
        self.tau = tau
        self.score_avg = 0
        self.best_score = 0

        self.actor = Actor(1,n_actions,D=4,name=actor_name)
        self.critic = Critic(1,n_actions,D=4,name=critic_name)
        self.target_actor = Actor(1,n_actions,D=4,name='targ_'+actor_name)
        self.target_critic = Critic(1,n_actions,D=4,name='targ_'+critic_name)

        # loads the convolutional layers from a pre-trained critic
        if transfer:
            temp = Actor(name='chckptn_supervised_actor')
            temp.load_checkpoint(load_as='chckptn_supervised_actor')
            # load every layer for critic
            self.critic.conv1.load_state_dict(temp.conv1.state_dict())
            for name,param in self.critic.named_parameters():
                if 'conv1' in name:
                    param.requires_grad = False

            # load last layers from PID_train
            self.actor.conv1.load_state_dict(temp.conv1.state_dict())
            for name,param in self.actor.named_parameters():
                if 'conv1' in name:
                    param.requires_grad = False
            
            del temp

            non_frozen_params = [p for p in self.actor.parameters() if p.requires_grad]
            self.actor_optim = Adam(params=non_frozen_params, lr=alpha)
            non_frozen_params = [p for p in self.critic.parameters() if p.requires_grad]
            self.critic_optim = Adam(params=non_frozen_params, lr=beta)
        else: 
            self.actor_optim = Adam(params=self.actor.parameters(), lr=alpha)
            self.critic_optim = Adam(params=self.critic.parameters(), lr=beta)

        self.update_network_params(tau=1) # hard copy

    def choose_action(self, state, evaluate=False):
        self.actor.eval()
        x,jnt_pos, jnt_goal = act_preprocessing(state,single_value=True)
        with torch.no_grad():
            action = self.actor.forward(x,jnt_pos, jnt_goal)

        if not evaluate:
            e = np.random.random()
            if e <= self.e:
                noise = self.enoise
            else:
                noise = self.noise
            action += torch.normal(torch.zeros_like(action).to(self.actor.device),noise)

        return action
    
    def update_network_params(self, tau=None):
        if tau == None:
            tau = self.tau

        critic_dict = self.critic.state_dict()
        actor_dict = self.actor.state_dict()
        target_critic_dict = self.target_critic.state_dict()
        target_actor_dict = self.target_actor.state_dict()

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() + \
                                (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_dict)

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() + \
                                (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_dict)

    def remember(self, state, weights, action, reward, new_state, done,t):
        self.memory.store_transition(state,weights,action,reward,new_state,done,t)

    def learn(self,batch=None,use_batch=False,use_data=False,data=None):
        if self.memory.mem_cntr < self.batch_size:
            return 0.0
        if use_data:
            state, action, weight, reward, new_state, done = data[0],data[1],data[3],data[4],data[5]
        elif use_batch:
            state, action, weight, reward, new_state, done = self.memory.sample_buffer(self.batch_size,use_batch=True,batch=batch)
        else:
            state, action, weight, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        new_x, new_jnt_err, new_jnt_dedt, w = act_preprocessing(new_state, weights)
        x, jnt_err, jnt_dedt, w, a = crit_preprocessing(state, weights, action)

        # target actions
        target_actions = self.target_actor.forward(new_x,new_jnt_err, new_jnt_dedt,w)
        # target critic value - value of next state (next state and next action)
        critic_value_ = self.target_critic.forward(new_x,new_jnt_err,new_jnt_dedt,w, target_actions)
        

        target=[]
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*(1-done[j]))
        target = torch.vstack(target)
        
        # update critic 
        self.critic_optim.zero_grad()
        critic_value = self.critic.forward(x,jnt_pos,jnt_goal,w,a)
        # critic_loss = F.l1_loss(critic_value, target)
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        mu = self.actor.forward(x,jnt_pos,jnt_goal,w)
        actor_loss = -1*self.critic.forward(x,jnt_pos,jnt_goal,w,mu).mean()
        # print('actor_loss',actor_loss)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_network_params()
        self.actor.eval()
        self.critic.eval()

        return critic_loss.item()

    def save_models(self, chckptn=False):
        print('saving models')
        if chckptn:
            str1 = save_as + self.actor.name
            str2 = save_as + self.critic.name
            str3 = save_as + self.target_actor.name
            str4 = save_as + self.target_critic.name
        else:
            str1,str2,str3,str4 = None,None,None,None
        self.actor.save_checkpoint(save_as=str1)
        self.critic.save_checkpoint(save_as=str2)
        self.target_actor.save_checkpoint(save_as=str3)
        self.target_critic.save_checkpoint(save_as=str4)

    def load_models(self):
        print('loading models')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def load_memory(self):
        self.memory = self.memory.load()



    # def act_preprocessing(self, state,single_value=False):
    #     if single_value:
    #         coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
    #         jnt_err = state[2]#.clone().detach()
    #         jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
    #     else:
    #         coords,feats = ME.utils.sparse_collate(state[0],state[1])
    #         jnt_err = state[2]#.clone().detach()
    #         jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda')

    #     x = ME.SparseTensor(coordinates=coords, features=feats.double(),device='cuda')
    #     return x, jnt_err

    # def crit_preprocessing(self, state, action, single_value=False):
    #     if single_value:
    #         coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
    #         jnt_err = state[2]#.clone().detach()
    #         jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
    #         a = torch.tensor(action,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
    #     else:
    #         coords,feats = ME.utils.sparse_collate(state[0],state[1])
    #         jnt_err = state[2]#.clone().detach()
    #         jnt_err = torch.tensor(state[2],dtype=torch.double,device='cuda')
    #         # action = action.clone().detach()
    #         # this line causes the thing not to learn
    #         a = torch.tensor(action,dtype=torch.double,device='cuda') 

    #     x = ME.SparseTensor(coordinates=coords, features=feats.double(),device='cuda')
    #     return x, jnt_err, a