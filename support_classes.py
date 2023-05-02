import random 
import numpy as np
from env_config import dt 
from rtree import index

class vertex(object):
    def __init__(self, th, t = 0, w=None, tau=None, J= None, reward=0, targ=None, id=0):
        self.th = th # (th1,th2,th3) tuple
        self.t = t
        self.reward = reward
        if isinstance(w, np.ndarray):
            self.w = w
            self.tau = tau
            self.J = J
        else:
            self.w = np.zeros_like(self.th)
            self.tau = np.zeros_like(self.th)
            self.J = np.zeros_like(self.th)
        self.targ = targ
        self.id = id
        self.tag = False

    def copy(self):
        new_v = vertex(th=self.th,t=self.t,w=self.w,tau=self.tau,J=self.J,reward=self.reward,targ=self.targ,id=self.id)
        return new_v


class Tree(object):
    def __init__(self, dims):
        '''
        Tree 
        dims: dimension of storage space
        X: search space
        V: joint-space spatial storage of vertex's
        E: dictionary of edges 
        '''
        p = index.Property()
        p.dimension = dims
        self.V = index.Index(interleaved=True, properties=p)
        self.V_count = 0
        self.E = {}