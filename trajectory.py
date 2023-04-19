import numpy as np
from env_config import dt, t_limit, tau_max

def Trajectory(traj):
    '''
    traj: list of tups with each entry (t, th, w) being a saved
    wave point along a path
    returns: an array of tau's from t=0 to t=t_final
    '''

    def tau(t, t1, a, b):
        return (6*a)*(t - t1) + 2*b

    def th(t, t1, a, b, c, d):
        t_ = t - t1
        return (a*t_**3)

    (t1,th1,w1, targ) = traj.pop()
    t = t1
    tau_list = []
    while len(traj) > 0:
        # th(t) = at**3 + bt**2 + c*t + d
        (t2, th2, w2, targ) = traj.pop()

        while t < t2 and t < t_limit/dt:
            tau_list.append(targ)
            t += 1
    
        t1 = t2
        t = t1
        th1 = th2
        w1 = w2
    
    return tau_list


def Trajectory_v2(jerk_arr, t_arr):
    '''
    jerk_arr = array of jnt_dxn joint jerk values, n=#of nodes in the path
    t_arr = the time steps of each of the nodes
    returns: 
        actions: an array of actions taken at every time step along the path
                 any entry that is np.inf was not changed during calculations
    '''
    t = 0
    r,c = jerk_arr.shape
    tf = int(t_arr[-1])
    actions = np.ones((tf,c)) * np.inf
    q_2dot = np.zeros(c)
    for i in range(0,r):
        J = jerk_arr[i,:]
        t2 = t_arr[i]
        while t < t2:
            q_2dot = J*dt + q_2dot
            q_2dot = np.clip(q_2dot,-tau_max,tau_max)
            actions[t,:] = q_2dot
            t+=1
    
    return actions
