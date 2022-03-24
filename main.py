from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
from reacher_moving_final import ReacherMovingFinalEnv

import train
import buffer

env = ReacherMovingFinalEnv()#gym.make('Reachermoving-v1')

MAX_EPISODES = 5000
MAX_STEPS = 200
MAX_BUFFER = MAX_EPISODES*MAX_STEPS
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
max_tank_array=[]
total_eout_array=[]

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    max_tank = 0.0
    trainer.reset()
    total_dist_reward=0.0
    total_energy_reward=0.0
    total_time_reward=0.0
    total_reward = 0.0
    total_eout =0.0
    actor_loss_total = 0.0
    critic_loss_total = 0.0
    total_steps = MAX_STEPS
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        total_dist_reward += info['reward_dist']
        total_energy_reward += info['reward_energy']
        total_time_reward += info['reward_time']
        total_eout +=info['e_out']
        total_reward += reward
        if max_tank<info['e_tank']:
            max_tank=info['e_tank']

        new_state = np.float32(new_observation)
            # push this exp in ram
        ram.add(state, action, reward, new_state, done)

        observation = new_observation

        # perform optimization
        critic_loss,actor_loss = trainer.optimize()
        critic_loss_total += critic_loss
        actor_loss_total += actor_loss
        if done:
            total_steps = r+1
            break
    print("energy reward: ",total_energy_reward,"  distance reward:",total_dist_reward,"  total reward:",total_reward," total time::",total_time_reward,"  critic loss:",critic_loss_total,"  actor loss:",actor_loss_total)
    print("total e_out:",total_eout)
    print("\n")
    print("\n")
    # check memory consumption and clear memory
    gc.collect()
    reward_list.append(total_reward)
    energy_density_array.append(total_eout)
    max_tank_array.append(max_tank)
    total_eout_array.append(total_eout)
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)
    if _ep%20 == 0 and _ep != 0:
        trainer.save_models(_ep)
        observation = env.reset()
        trainer.reset()
        total_dist_reward=0.0
        total_energy_reward=0.0
        total_time_reward=0.0
        total_reward = 0.0
        for r in range(MAX_STEPS):
            env.render()
            state = np.float32(observation)
            action = trainer.get_exploitation_action(state)
            new_observation, reward, done, info = env.step(action)
            
            total_dist_reward += info['reward_dist']
            total_energy_reward += info['reward_energy']
            total_time_reward += info['reward_time']
            total_eout += info['e_out']
            total_reward += reward
            
            observation = new_observation
            if done:
                break
        print('testing data')
        print("energy reward: ",total_energy_reward," time reward:",total_time_reward,"  distance reward:",total_dist_reward,"  total reward:",total_reward)
        print("total_eout:",total_eout)
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

reward_np = np.array(avg_reward)
np.save('/home/sid/source/thesis/Data/reward',reward_np)

edensity_np = np.array(energy_density_array)
np.save('/home/sid/source/thesis/Data/edensity',edensity_np)

maxtank_np = np.array(max_tank_array)
np.save('/home/sid/source/thesis/Data/maxtank',maxtank_np)


totaleout_np = np.array(total_eout_array)
np.save('/home/sid/source/thesis/Data/totaleout',totaleout_np)



fig,axs =plt.subplots(3)
axs[0].plot(avg_reward)
axs[1].plot(avg_edensity)
axs[2].plot(total_eout_array)
axs[2].plot(max_tank_array)
plt.show()
print('Completed episodes')