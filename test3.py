import warnings
warnings.filterwarnings('ignore')
from sparse_tnsr_replay_buffer import ReplayBuffer  

# name_list = 
name_list = ['rrt_data1', 'rrt_data2', 'rrt_data3', 'rrt_data4', 
            'rrt_data5', 'rrt_data6', 'rrt_data7', 'rrt_data8',
            'rrt_data9']
buffer = ReplayBuffer(file='train_data')
for name in name_list:
    temp = ReplayBuffer(file=name)
    temp = temp.load()
    buffer.add_data([temp])
    print('adding',name,buffer.mem_cntr)
buffer.save()