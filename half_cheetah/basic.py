from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt

import train
import buffer

env = gym.make('Reacher-v0')
# env = gym.make('Pendulum-v0')

MAX_EPISODES = 5000
MAX_STEPS = 100
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

for _ep in range(MAX_EPISODES):
    observation = env.reset()

    trainer.reset()
    total_dist_reward=0.0
    total_ctrl_reward=0.0
    total_reward = 0.0
    actor_loss_total = 0.0
    critic_loss_total = 0.0
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        #env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)
        # if _ep%5 == 0:
        #   # validate every 5th episode
        #   action = trainer.get_exploitation_action(state)
        # else:
        #   # get action based on observation, use exploration policy here
        #   action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        total_dist_reward += info['reward_dist']
        total_ctrl_reward += info['reward_ctrl']
        total_reward += reward

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        #   continue

        new_state = np.float32(new_observation)
            # push this exp in ram
        ram.add(state, action, reward, new_state, done)

        observation = new_observation

        # perform optimization
        critic_loss,actor_loss = trainer.optimize()
        critic_loss_total += critic_loss
        actor_loss_total += actor_loss
        if done:
            #print("target reached")
            break
    print("Ctrl reward:",total_ctrl_reward,"  distance reward:",total_dist_reward,"  total reward:",total_reward,"  critic loss:",critic_loss_total,"  actor loss:",actor_loss_total)
    # check memory consumption and clear memory
    gc.collect()
    reward_list.append(total_reward)
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)
    if _ep==10 or (_ep%100 == 0 and _ep!=0):
        trainer.save_models(_ep)
        
        observation = env.reset()
        trainer.reset()
        total_dist_reward=0.0
        total_ctrl_reward=0.0
        total_reward = 0.0
        for r in range(MAX_STEPS):
            env.render()
            state = np.float32(observation)
            action = trainer.get_exploitation_action(state)
            new_observation, reward, done, info = env.step(action)
            
            total_dist_reward += info['reward_dist']
            total_ctrl_reward += info['reward_ctrl']
            total_reward += reward
            
            observation = new_observation
            if done:
                break
        print('testing data')
        print(" ctrl reward:",total_ctrl_reward,"  distance reward:",total_dist_reward,"  total reward:",total_reward)
        
avg_reward=[]
for i in range(len(reward_list)-99):
    av_10 = 0.0
    for j in range(100):
        av_10 += reward_list[i+j]
    avg_reward.append(av_10/100)
plt.plot(avg_reward)
plt.show()
print('Completed episodes')