from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
from env.reacher1D import Reacher1DEnv

import train
import buffer

env = Reacher1DEnv()

MAX_EPISODES = 10000
MAX_STEPS = 200
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

reward_list=[]
energy_density_array=[]

for _ep in range(MAX_EPISODES):
    observation = env.reset()

    trainer.reset()
    total_tank_reward=0.0
    total_energy_reward=0.0
    total_time_reward=0.0
    total_reward = 0.0
    total_eout =0.0
    actor_loss_total = 0.0
    critic_loss_total = 0.0
    total_steps = MAX_STEPS
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        #env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        total_tank_reward += info['reward_tank']
        total_energy_reward += info['reward_energy']
        #total_time_reward += info['reward_time']
        total_eout += info['e_out']
        total_reward += reward

        new_state = np.float32(new_observation)
        ram.add(state, action, reward, new_state, done)

        observation = new_observation

        # perform optimization
        critic_loss,actor_loss = trainer.optimize()
        critic_loss_total += critic_loss
        actor_loss_total += actor_loss
        if done:
            total_steps = r+1
            break
    print("energy reward: ",total_energy_reward,"  tank reward:",-total_tank_reward,"  total reward:",total_reward," total eout",total_eout,"  critic loss:",critic_loss_total,"  actor loss:",actor_loss_total)
    print("\n")
    gc.collect()
    reward_list.append(total_reward)
    energy_density_array.append(total_eout/total_steps)
    if _ep%50 == 0 and _ep != 0:
        trainer.save_models(_ep)
        observation = env.reset()
        trainer.reset()
        total_tank_reward=0.0
        total_energy_reward=0.0
        total_reward = 0.0
        for r in range(MAX_STEPS):
            env.render()
            state = np.float32(observation)
            action = trainer.get_exploitation_action(state)
            new_observation, reward, done, info = env.step(action)
            print("e_tank:",new_observation[7])

            total_tank_reward += info['reward_tank']
            total_energy_reward += info['reward_energy']
            #total_time_reward += info['reward_time']
            total_reward += reward
            
            observation = new_observation
            if done:
                break
        print('testing data')
        print("energy reward: ",total_energy_reward,"  tank reward:",-total_tank_reward,"  total reward:",total_reward)

avg_reward=[]
avg_edensity=[]
for i in range(len(reward_list)-99):
    av_100 = 0.0
    for j in range(99):
        av_100 += reward_list[i+j]
    avg_reward.append(av_100/100)

for i in range(len(energy_density_array)-99):
    av_100 = 0.0
    for j in range(99):
        av_100 += energy_density_array[i+j]
    avg_edensity.append(av_100/100)

all_reward_np = np.array(avg_reward)
np.save('./plot_data/reward',all_reward_np)

fig,axs =plt.subplots(2)
axs[0].plot(avg_reward)
axs[1].plot(avg_edensity)
plt.show()
print('Completed episodes')