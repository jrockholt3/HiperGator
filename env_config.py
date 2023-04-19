import numpy as np

dt = 0.016# time step
t_limit = 6.0 # time limit in seconds
thres = (5*np.pi/180)*np.ones(5,dtype=float) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
prox_thres = .05 # proximity threshold - 5 cm
min_prox = np.inf*-1
vel_prox = np.inf*-1

# Controller gains
tau_max = 1.5 #rad/s^2
damping = .8*tau_max
Z = damping 
P = 10
D = 5
P_v = 10.0
D_v = 5.0
jnt_vel_max = 1.5 # rad/s
j_max = tau_max/dt  # maximum allowable jerk on the joints

pi = np.pi
jnt_max = np.array([np.pi, .9*np.pi, .9*np.pi, np.pi, .9*np.pi])
jnt_min = np.array([-np.pi, -.9*np.pi, -.9*np.pi, -np.pi, -.9*np.pi])