from Robot_5link_Env import RobotEnv,gen_obs_pos
from Robot_5link import S,a,l
from env_config import *
from support_classes import vertex, Tree
import numpy as np 
# import matplotlib as plt
# from mpl_toolkits import mplot3d
from time import time 
from optimized_functions_5L import nxt_state, reward
from optimized_functions import angle_calc

class RRT_star():
    def __init__(self, start, goal, max_samples, r, d, thres, n, steps, env:RobotEnv):
        self.n = n
        self.steps = steps
        self.thres = thres
        self.start = start
        self.goal = goal
        self.max_samples = max_samples
        self.sample_count = 0
        self.r = r
        self.d = d
        self.tree = Tree(dims=5)
        self.edge_count = 0
        self.stop = False
        self.jnt_min = jnt_min
        self.jnt_max = jnt_max
        self.env = env
        self.obs_dict = gen_obs_pos(env.objs)

    def add_vertex(self, v:vertex):
        """
        insert vertex into tree
        """
        self.tree.V.insert(v.id, v.th, obj=v)
        self.tree.V_count += 1
        self.sample_count += 1

    def add_edge(self, child:vertex, parent:vertex):
        # connect parent to child
        self.edge_count += 1
        child.id = self.edge_count
        self.tree.E[child.id] = parent
        return child 
    
    def init_root(self, start:vertex):
        self.tree.E[start.id] = start

    def sample(self):
        th = np.random.uniform(self.jnt_min, self.jnt_max)
        return th

    def nearby(self, th, n):
        '''
        th: robot pose
        n: number of neighbors to return
        return list of neighbors as vertex
        '''
        return self.tree.V.nearest(th, num_results=n,objects="raw")
    
    def get_nearest(self, th):
        """
        return nearest vertex
        """
        return next(self.nearby(th,1))
    
    def get_win_radius(self, center, r):
        top = (center[0]+r,center[1]+r,center[2]+r,center[3]+r,center[4]+r) 
        bottom = (center[0]-r,center[1]-r,center[2]-r,center[3]-r,center[4]-r)
        coords = (bottom[0], bottom[1], bottom[2], bottom[3], bottom[4], 
                top[0], top[1], top[2], top[3], top[4])
        return list(self.tree.V.intersection(coords,objects='raw'))
    
    def steer(self, th1, th2):
        d = self.d
        start, end = np.array(th1),np.array(th2)
        norm = np.linalg.norm(end - start)
        if norm >= d:
            v = (end - start) / norm
            steered_ptn = angle_calc(start + v*d)
            return tuple(steered_ptn)
        else:
            return th2
    
    def get_reachable(self, v, targ):
        if v.t < t_limit/dt:
            targ_i = self.steer(v.th, targ)
            th_fin, w_fin, reward, t_fin, flag = self.env.env_replay(v, targ_i, self.obs_dict, self.steps)
            return vertex(tuple(th_fin), t_fin, w_fin, reward, targ=targ_i), flag
        else:
            return None, False

    def get_reachable_v2(self, v, targ):
        # calculates a reachable node give the target and current node
        # by setting jerk
        if v.t < t_limit/dt:
            targ_i = self.steer(v.th, targ)
            q_ = np.array(targ_i)
            t,q,q_dot,q_2dot = v.t,np.array(v.th),v.w,v.tau
            J = (q_ - (q_2dot*dt**2/2 + q_dot*dt + q)) / (dt**3/6)
            J = np.clip(J, -1*j_max, j_max) # clipping for maximum jerk
            score = 0
            for i in range(self.steps):
                q_2dot = J*dt + q_2dot
                q_2dot = np.clip(q_2dot, -1*tau_max, tau_max)
                temp = nxt_state(self.obs_dict[t],q,q_dot,q_2dot,a,l,S)
                q = temp[0:q.shape[0],0]
                q_dot = temp[0:q.shape[0],1]
                t = t+1
                score += reward()
            d = np.linalg.norm(q - np.array(v.th))
            if d >= self.thres:
                v = vertex(th=tuple(q),t=t,w=q_dot,tau=q_2dot,J=J,reward=score,targ=targ_i)
                return v, True
            else:
                v = vertex(None)
                return v, False
        else:
            v = vertex(None)
            return v, False

    def reconstruct_path(self, curr, start, goal):
        '''
        start: (th1, th2, th3)
        goal: (th1, th2, th3)
        '''
        # path = [goal]
        path = []
        traj = []
        # path.append(goal)
        if start == goal:
            print('reconstruct_path: start and goal are eqaul')
            return path
        while not curr.id == 0:
            # path.append((curr.t,curr.th,curr.targ))
            path.append(curr.t)
            traj.append(curr.J)
            curr = self.tree.E[curr.id]
        # path.append()

        path.reverse() 
        traj.reverse()
        
        return path, traj
    
    def reward_calc(self, leaf:vertex):
        curr_v = leaf
        reward = 0
        while not curr_v.th == self.start:
            reward += curr_v.reward
            curr_v = self.tree.E[curr_v.id]
        return reward
    
    def add_highest_reward(self, pv_pairs):
        '''
        add the parent-leaf pair with the highest
        reward
        '''
        flag = False
        best_reward = -np.inf
        for tup in pv_pairs:
            p,v = tup[0],tup[1]
            d = np.linalg.norm(np.array(p.th) - np.array(v.th))
            reward = self.reward_calc(p) + v.reward
            if d >= self.thres and reward > best_reward:
                flag = True
                best_reward = reward
                parent = p
                v_best = v
        
        if flag:
            child = self.add_edge(v_best, parent)
            self.add_vertex(child)
            return child, flag
        else:
            return None, flag
        
    def can_connect_to_goal(self):
        # check if we can connect
        v_nearest = self.get_nearest(self.goal)
        if self.goal in self.tree.E and v_nearest.th in self.tree.E[self.goal]:
            # goal is already connected 
            return True
        if np.linalg.norm(np.array(v_nearest.th) - self.goal) <= self.r:
            # v_goal = vertex(th=self.goal, t=v_nearest.t+1)
            # self.add_vertex(v_goal)
            # self.add_edge(v_goal, v_nearest)
            return v_nearest, True
        return None, False
        
    def rrt_search(self):
        v = vertex(self.start)
        self.init_root(v)
        self.add_vertex(v)
        converged = False
        loop_count = 0
        t_list = []
        traj=[]
        v = self.tree.E[0]
        for i in range(100):
            th_new = self.sample()
            v_reached, flag = self.get_reachable_v2(v,th_new)
            if flag:
                child = self.add_edge(v_reached,v)
                self.add_vertex(child)
        
        while not converged:
            # sample a new node
            if loop_count%10 == 0:
                th_new = self.goal # bias
            else:
                th_new = self.sample() # explore 
            # find n nearest nodes
            near = self.nearby(th_new, self.n)
            # try to reach new node from nearby nodes
            v_new = []
            for v in near:
                # v = self.recalc_path(v) # update v's position
                v_reached,flag = self.get_reachable_v2(v, th_new)
                if flag:
                    v_new.append((v,v_reached))
            # add the node with the best reward
            if len(v_new) != 0:
                v, v_added_flag = self.add_highest_reward(v_new)
           
            loop_count += 1
            if loop_count%100 == 0:
                v_near = self.get_win_radius(self.goal, self.r)
                if len(v_near) > self.n:
                    print('converaged',len(v_near))
                    r_best = -np.inf
                    for v in v_near:
                        r_i = self.reward_calc(v)
                        if r_i > r_best:
                            v_best = v.copy()
                            r_best = r_i
                    converged = True
                    t_list,traj = self.reconstruct_path(v_best, self.start, self.goal)
                
            if loop_count>=self.max_samples:
                v, converged = self.can_connect_to_goal()
                if converged:
                    t_list,traj = self.reconstruct_path(v, self.start, self.goal)
                else:
                    print('failed to converge')
                    converged = True
        
        print("edge_count",self.edge_count)
        return t_list, traj
    
    def plot_graph(self, every=10, add_path=False, path=None):
        # entres = np.empty(len(self.tree.E),dtype=tuple)
        # for k in self.tree.E.keys():
        #     parent = self.tree.E[k]
        #     entres[parent.id] = parent.th

        th_arr = []
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        i = 0
        for k in self.tree.E.keys():
            if i%every == 0:
                parent = self.tree.E[k]
                arr = np.array(parent.th)
                th_arr.append(arr)
                # child_th = entres[k]
                # # parent = self.tree.E[k]
                # xx = np.array([parent.th[0], child_th[0]])
                # yy = np.array([parent.th[1], child_th[0]])
                # zz = np.array([parent.th[2], child_th[0]])
                # ax.plot3D(xx,yy,zz,'k',alpha=.1)

        th_arr = np.vstack(th_arr)
        xx = th_arr[:,0]
        yy = th_arr[:,1]
        zz = th_arr[:,2]

        ax.scatter3D(xx,yy,zz,alpha=.1)
        ax.scatter3D(self.start[0], self.start[1], self.start[2], 'r')
        ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], 'g')

        if add_path:
            xx,yy,zz = [],[],[]
            for th in path:
                xx.append(th[0])
                yy.append(th[1])
                zz.append(th[2])

            ax.plot3D(xx,yy,zz, 'r')

        plt.show()


# env = RobotEnv()
# start = tuple(env.start)
# goal = tuple(env.goal)
# steps = 5
# thres = np.sqrt(5*.003)**2 * (steps/25)
# n = 25
# r = np.linalg.norm(jnt_vel_max*dt*steps*np.ones(5)/3)
# d = np.linalg.norm(jnt_vel_max*dt*steps*1.5*np.ones(5))
# max_samples = 5000
# rrt = RRT_star(start,goal,max_samples,r,d,thres,n,steps,env)
# path,traj = rrt.rrt_search()
# rrt.plot_graph(every=2.0,add_path=True,path=path)