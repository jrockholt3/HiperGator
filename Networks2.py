import os 
import torch.nn as nn
from torch.optim import Adam
import MinkowskiEngine as ME
import numpy as np
# from Object_v2 import rand_object, Cylinder
import torch
from time import time 
from sparse_tnsr_replay_buffer import ReplayBuffer
from env_config import *

conv_out1 = 32
conv_out2 = 64
conv_out3 = 64
conv_out4 = 512
linear_out = 512
dropout = 0.2
k1 = (3,4,4,4) # kernel shape
s1 = (2,2,2,2) # stride 
k2 = (3,4,4,4)
s2 = (2,2,2,2)
k3 = (1,2,2,2)
s3 = (1,2,2,2)
k4 = (1,9,9,6)
s4 = (1,1,1,1)

kg1= ME.KernelGenerator(
    kernel_size = k1,
    stride = s1,
    region_type=ME.RegionType.HYPERCROSS,
    dimension=4
)
kg2= ME.KernelGenerator(
    kernel_size = k2,
    stride = s2,
    region_type=ME.RegionType.HYPERCROSS,
    dimension=4
)
kg3= ME.KernelGenerator(
    kernel_size = k3,
    stride = s3,
    region_type=ME.RegionType.HYPERCROSS,
    dimension=4
)
kg4= ME.KernelGenerator(
    kernel_size = k4,
    stride = s4,
    region_type=ME.RegionType.HYPERCROSS,
    dimension=4
)



class ActorCritic(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat, jnt_dim, D, name, chckpt_dir = 'tmp', device='cuda'):
        self.D = D
        super(Actor, self).__init__(self.D)
        self.name = name 
        self.chckpt_dir = chckpt_dir
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg')
        
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=k1,
                stride=s1,
                bias=True,
                kernel_generator=kg1,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out1).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=k2,
                stride=s2,
                bias=True,
                kernel_generator=kg2,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out2).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out3,
                kernel_size=k3,
                stride=s3,
                bias=True,
                kernel_generator=kg3,
                dimension=D).double(),
            ME.MinkowskiInstanceNorm(conv_out3).double(),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiConvolution(in_channels=conv_out3,
                out_channels=conv_out4,
                kernel_size=k4,
                stride=s4,
                bias=True,
                kernel_generator=kg4,
                dimension=D).double(),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
            ME.MinkowskiGlobalMaxPooling(),
            ME.MinkowskiInstanceNorm(conv_out4),
            ME.MinkowskiSELU().double(),
            ME.MinkowskiDropout(dropout).double()
        )
        self.actor = nn.Sequential(
            nn.Linear(conv_out4+2*jnt_dim+3,linear_out,bias=True).double(), # first liner layer
            nn.LayerNorm(linear_out).double(),
            nn.SELU().double(),
            nn.Dropout(dropout).double(),
            nn.Linear(linear_out,jnt_dim,bias=True).double(), # final layer 
            nn.Tanh().double()
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out4+3*jnt_dim+3,linear_out,bias=True).double(),
            nn.LayerNorm(linear_out).double(),
            nn.SELU().double(),
            nn.Dropout(dropout).double(),
            nn.Linear(linear_out,1,bias=True).double()
        )
        

        # self.action_var = torch.full((n_actions, ), action_std*action_std).to(device)
        self.device = device
        self.to(self.device)
        # if device == "cuda":
        #     self.conv1.cuda(0)
        #     self.norm.cuda(1)
        #     self.dropout1.cuda(1)
        #     self.linear.cuda(1)
        #     self.dropout2.cuda(1)
        #     self.out.cuda(1)
        #     # devcie = device + ":0"
        # else:
        #     self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y


    def forward(self,x:ME.SparseTensor,jnt_err,jnt_dedt,weights,actions):
        x = self.conv1(x)
        x = self.to_dense_tnsr(x)
        act_in = torch.cat((jnt_err,jnt_dedt,weights,x),dim=1)
        crit_in = torch.cat((jnt_err,jnt_dedt,weights,actions,x),dim=1)
        act_out = self.actor(act_in)
        crit_out = self.critic(crit_it)
        return act_out, crit_out 

    def save_checkpoint(self, save_as=None):
        if isinstance(save_as,str):
            name = save_as
            file_path = os.path.join(self.chckpt_dir,save_as+'_ddpg')
        else:
            file_path = self.file_path

        print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, name=None):
        if isinstance(name,str):
            tmp_file_path = os.path.join(chckpt_dir, name)
        else:
            tmp_file_path = self.file_path
        # print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(tmp_file_path))
