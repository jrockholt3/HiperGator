from numba import njit, int32, float64
from optimized_functions import T_1F, T_ji, T_inverse, calc_clip_vel, calc_jnt_err, angle_calc, PDControl
import numpy as np
from env_config import *
from env_config import damping as Z
from Robot_5link import shift, get_transforms, points

# @njit((float64[:])(int32,float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
# def asb_link_i(link, obj_pos, th, a, l, S):
#     """
#     inputs: link = body #
#         th = jnt angle [th1, th2, ... thn]
#         a = twist angle [aph1, aph2 ... aphn]
#         l = link length [l1, l2, .. ln]
#         S = joint offsets [S1, S2, ... Sn]
#     return: the objects position asb link # link
#     """
#     if th.shape[0] < link:
#         print('asb_link_i: not enough links, link=', link)
#         return np.array([np.nan, np.nan, np.nan], dtype=float64)
#     T_arr = get_transforms(th,S,a,l)
#     # T = T_1F(th[0],S[0])
#     # for i in range(1, link):
#     #     T = T@T_ji(th[i],a[i-1],l[i-1],S[i])
#     for i in range()
#     inv_T = T_inverse(T)
#     vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
#     return inv_T@vec

@njit((float64[:,:])(float64), nogil=True)
def Rot_Z(th):
    c = np.cos(th)
    s = np.sin(th)
    T = np.array([[  c,  -s, 0.0, 0.0],
                  [  s,   c, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return T.T

@njit((float64)(float64[:],float64[:],float64[:],float64[:],float64[:]), nogil=True)
def proximity(obj_pos, th, a, l, S):
    '''
    input 
        obj_pos = [x,y,z] of obj
        '''
    step_size = .05 # step size along robot arm in meters
    point_arr = points(th,S,a,l)
    O2 = point_arr[0:3,1]
    O3 = point_arr[0:3,2]
    O5 = point_arr[0:3,3]
    Ptool_f = point_arr[0:3,4] 
    u23 = np.linalg.norm(O3-O2)
    u23_f = (O3 - O2)/u23
    u35 = np.linalg.norm(O5-O3)
    u35_f = (O5 - O3)/u35
    u5t = np.linalg.norm(Ptool_f-O5)
    u5t_f = (Ptool_f - O5)/u5t
    max_size = int(np.ceil((u23+u35+u5t)/step_size))
    proxs = np.ones(max_size) * np.inf
    s = 0.0
    for i in range(max_size):
        if s < u23:
            u_i = s*u23_f + O2
        elif s < u23 + u35:
            u_i = u35_f * (s - u23) + O3
        elif s <= u23 + u35 + u5t:
            u_i = u5t_f * (s - u23+u35) + O5
            
        prox_i = np.linalg.norm(obj_pos - u_i)
        proxs[i] = prox_i
        s += step_size
    prox = np.min(proxs)

    return prox

@njit((float64[:,:])
      (float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]),
      nogil=True)
def nxt_state(obj_pos, th, w, tau, a, l, S):
    '''
    obj_pos = [pos1, pos2,...]
    '''
    prox_arr = np.zeros(obj_pos.shape[1],dtype=float)
    for i in range(0, obj_pos.shape[1]):
        prox_arr[i] = proximity(obj_pos[:,i], th, a, l, S)
    prox = np.min(prox_arr)

    if prox <= vel_prox:
        vel_clip = calc_clip_vel(prox)
    else:
        vel_clip = jnt_vel_max

    if prox <= min_prox:
        paused = True
    else:
        paused = False

    if not paused:
        nxt_w = (tau - Z*w)*dt + w
        if np.any(np.abs(nxt_w) > vel_clip):
            nxt_w = np.clip(nxt_w, -vel_clip, vel_clip)
            tau = (nxt_w - w)/dt + Z*w
        
        nxt_th = (tau - Z*w)*dt**2/2 + w*dt + th
    else:
        nxt_w = np.zeros(w.shape[0], dtype=float)
        nxt_th = th

    if np.any(nxt_th >= jnt_max) or np.any(nxt_th <= jnt_min):
        nxt_w[nxt_th >= jnt_max] = 0
        nxt_w[nxt_th <= jnt_min] = 0
        nxt_th[nxt_th >= jnt_max] = jnt_max[nxt_th >= jnt_max]
        nxt_th[nxt_th <= jnt_min] = jnt_min[nxt_th <= jnt_min]

    package = np.zeros((6,2),dtype=float)
    for i in range(0,nxt_th.shape[0]):
        package[i,0] = nxt_th[i]

    for i in range(0,nxt_w.shape[0]):
        package[i,1] = nxt_w[i]

    package[-1,0] = prox
    return package


def reward():
    return -1