import pickle
import matplotlib.pyplot as plt
import numpy as np

file = open('action_mem.pkl','rb')
actions = pickle.load(file)