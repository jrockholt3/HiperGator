import matplotlib.pyplot as plt
import csv 
import numpy as np

def plot_file(file,dt,title):
    reader = csv.reader(file)
    row_list = []
    for row in reader:
        temp = []
        if len(row) == 5:
            for x in row:
                temp.append(float(x))
            row_list.append(temp)

    row_arr = np.vstack(row_list)
    # print(row_arr[0,:])


    fig = plt.figure()
    plt.plot(np.arange(row_arr.shape[0])*dt, row_arr)
    plt.title(title+' Position')

    fig = plt.figure()
    q_dot = np.diff(row_arr, axis=0)/.016
    plt.plot(np.arange(q_dot.shape[0])*dt, q_dot)
    plt.title(title+' First Deriv')

    fig = plt.figure()
    q_2dot = np.diff(q_dot, axis=0)/.016
    plt.plot(np.arange(q_2dot.shape[0])*dt, q_2dot)
    plt.title(title+' 2nd Deriv')

# fig = plt.figure()
dt = 1/500
file = open('traj_out.csv')
plot_file(file,dt,'Traj')
dt = 1/500
file = open('goal_recieved.csv')
plot_file(file,dt,'Goal')
plt.show()