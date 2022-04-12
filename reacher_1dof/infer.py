from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
import argparse
import matplotlib.animation as animation
import time

import train
import buffer
from env.reacher1D import Reacher1DEnv

env = Reacher1DEnv()


MAX_BUFFER =10
MAX_timestep=200
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

pause = True
energy_and_action=[]
time_step=[]

def simData():
    total_dist_reward=0.0
    total_energy_reward=0.0
    total_time_reward=0.0
    total_reward = 0.0
    observation = env.reset()
    for r in range(MAX_timestep):
        env.render()
        state = np.float32(observation)
        action = trainer.get_exploitation_action(state)
        new_observation, reward, done, info = env.step(action)
        yield info['e_tank'],r,action
        observation = new_observation
        if done==True:
            break
        time.sleep(0.01)
def onClick(event):
    global pause
    pause ^= False
def simPoints(simData):
    x, t, act = simData[0], simData[1], simData[2]
    temp = [x,act[0]]
    energy_and_action.append(temp)
    time_step.append(t)
    plt.cla()
    plt.plot(time_step,energy_and_action)
    return

trainer.load_models(8000)
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], 'bo', ms=10)
ax.set_ylim(0, 40)
ax.set_xlim(0, 500)

time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
fig.canvas.mpl_connect('button_press_event', onClick)

ani = animation.FuncAnimation(fig, simPoints, simData, blit=False, interval=10, repeat=False)

plt.show()





