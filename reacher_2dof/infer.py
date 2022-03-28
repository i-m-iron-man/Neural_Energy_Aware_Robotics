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
from reacher_moving_final import ReacherMovingFinalEnv

env = ReacherMovingFinalEnv()
plt.rcParams.update({'font.size': 32})

MAX_BUFFER =10
MAX_timestep=1000
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

pause = False
energy=[]
time_step=[]
total_eout=[]
cumulutive=[]

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
        yield info['e_tank'],r,info['e_out']
        observation = new_observation
        if done==True:
            break
        time.sleep(0.01)
def onClick(event):
    global pause
    pause ^= True
def simPoints(simData):
    x, t, e_out = simData[0], simData[1], simData[2]
    energy.append(x)
    time_step.append(t)
    total_eout.append(e_out)
    cumulutive.append(sum(total_eout))
    plt.cla()
    plt.plot(time_step,energy,label="energy in tank")
    plt.plot(time_step,cumulutive,label="total energy")
    plt.xlabel('time')
    plt.ylabel('energy in joules')
    plt.legend()
    print(sum(total_eout))
    return

trainer.load_models(4800)
fig = plt.figure()
ax = fig.add_subplot(111)
#line, = ax.plot([], [], 'bo', ms=10)
ax.set_ylim(0, 40)
ax.set_xlim(0, 500)
ax.set_xlabel('time')
ax.set_ylabel('energy in joules')

time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
fig.canvas.mpl_connect('button_press_event', onClick)

ani = animation.FuncAnimation(fig, simPoints, simData, blit=False, interval=10, repeat=False)

plt.show()





