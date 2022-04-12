from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
import math

import train
import buffer
from env.half_cheetah_dynamic_tank import HalfCheetahDynamictankEnv

env = HalfCheetahDynamictankEnv()

MAX_EPISODES = 60000
MAX_STEPS = 200
MAX_BUFFER = 20000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

reward_list_train=[]

steps_list=[]
all_reward=[]

def variance(e_tank,e_tank_avg):
    vari = 0
    for i in e_tank:
        vari += (i-e_tank_avg)*(i-e_tank_avg)
    vari = vari/len(e_tank)
    return math.sqrt(vari)

#decide by howmuch the tank should be shifted in the beginning of each episode
def tank_decide():
    global steps_list
    if len(steps_list)>10:
        last_10 = steps_list[-10:]
        avg = sum(last_10)/10
        print("avg starting energy of last 10 episodes: ",avg)
        if avg < 60:
                return 20
        elif avg > 150:
            return -20
        else:
            return 0
    else:
        return 0




for _ep in range(MAX_EPISODES):
    energy_change = tank_decide()
    env.tank_change(energy_change)
    observation = env.reset()
    trainer.reset()
    total_reward = 0.0
    total_energy_reward=0.0
    total_survive_reward=0.0
    actor_loss_total = 0.0
    critic_loss_total = 0.0
    total_steps = MAX_STEPS
    e_out_total = 0.0
    e_tank_list = []
    final_dist = 0.0
    flag=False
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        #env.render()
        state = np.float32(observation)
        action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        e_out_total += info['e_out']
        e_tank_list.append(info['e_tank'])
        total_energy_reward += info['reward_energy']
        total_survive_reward += info['reward_survive']
        all_reward.append(info['reward_energy'])

        final_dist = new_observation[0]

        total_reward += reward

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
        if(len(all_reward)>=1000000):
            flag=True
            break
    gc.collect()
    reward_list_train.append(total_reward)
    
    if final_dist==0.0:
        final_dist=0.00001

    steps_list.append(total_steps)
    energy_density_distance = e_out_total/final_dist

    energy_density_time = e_out_total/total_steps
    
    energy_tank_avg = sum(e_tank_list)/len(e_tank_list)
    
    energy_tank_std_dev = variance(e_tank_list,energy_tank_avg)

    print("total energy reward:",total_energy_reward,"total time reward:",2*total_survive_reward,"  total reward:",total_reward,"  critic loss:",critic_loss_total,"  actor loss:",actor_loss_total)
    print("final_dist: ",final_dist)
    print("energy_density_distance: ", energy_density_distance)
    print("energy_density_time: ",energy_density_time)
    print("energy_tank_avg: ",energy_tank_avg)
    print("energy tank std dev: ",energy_tank_std_dev)
    print("\n")
    
    if _ep%500 == 0 and _ep != 0:
        trainer.save_models(_ep)
        observation = env.reset()
        trainer.reset()
        total_steps = MAX_STEPS
        total_reward = 0.0
        e_out_total =0.0
        final_dist=0.0
        e_tank_list = []
        for r in range(MAX_STEPS):
            env.render()
            state = np.float32(observation)
            action = trainer.get_exploitation_action(state)
            new_observation, reward, done, info = env.step(action)
            
            total_reward += reward
            e_out_total += info['e_out']
            e_tank_list.append(info['e_tank'])
            final_dist = new_observation[0]
            observation = new_observation
            if done:
                total_steps=r+1
                break
        energy_density_distance = e_out_total/final_dist

        energy_density_time = e_out_total/total_steps

        energy_tank_avg = sum(e_tank_list)/len(e_tank_list)
    
        energy_tank_std_dev = variance(e_tank_list,energy_tank_avg)
        
        print("testing...")
        print("total reward:",total_reward)
        print("final_dist: ",final_dist)
        print("energy_density_distance: ", energy_density_distance)
        print("energy_density_time: ",energy_density_time)
        print("energy_tank_avg: ",energy_tank_avg)
        print("energy tank std dev: ",energy_tank_std_dev)
        print("\n")
        if flag:
            break



all_reward_np = np.array(all_reward)




np.save('./plot_data/all_reward',all_reward_np)


print("end")
