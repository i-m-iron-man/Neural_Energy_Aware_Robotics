import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import random
import os

cwd = os.getcwd()

class ReacherMovingFinalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    max_energy = 10.0 # maximum alowed energy in the tank
    ein_coeff = 15 # constant map between reward and energu in0

    e_tank = 7 # initializing tank at energy < max_energy
    e_tank_old =  e_tank

    e_out_vec=[0.0,0.0] # list containing the energy going out from the 2 motors
    energy_going_out=0.0 # initializing total energy going out at a time step to 0
    energy_coming_in=0.0 # initialixing total energy coming in at a time step to 0

    theta = [0.0,0.0] # position of the 2 DOF
    theta_old =[0.0,0.0]
    init_dist=0.0
    target_com=[0.0,0.0,0.0]

    time_step = 0.05
    initial_target_radius= 0.0
    initial_target_angle=0.0
    angular_vel =0.0

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, cwd+'/reacher_no_box.xml', 2)

    # method to calculate energy out
    def energy_out(self,action):
        d_theta = self.theta-self.theta_old
        self.e_out_vec = action*d_theta
        if self.e_out_vec[0]<=0:
            self.e_out_vec[0] = action[0]*action[0]
        if self.e_out_vec[1]<=0:
            self.e_out_vec[1] = action[1]*action[1]
        self.energy_going_out = sum(self.e_out_vec)
        self.e_tank -= self.energy_going_out

    # method to calculate energy in
    def energy_in(self,prev_dist,next_dist):
        e_in = self.ein_coeff*(prev_dist-next_dist)
        self.energy_coming_in = e_in
        e_in = max(0.0,e_in)
        e_in = min(e_in,self.max_energy-self.e_tank_old)
        self.e_tank += e_in

    # method to calculate distance between target and end effector
    def calc_dist(self):
        vec = self.get_body_com("fingertip")[:2]-self.target_com[:2]
        return  np.linalg.norm(vec)
    
    # method to move the target with the angular vel
    def update_goal(self):
        self.initial_target_angle += self.angular_vel*self.time_step
        x_pos = self.initial_target_radius*math.sin(self.initial_target_angle)
        y_pos = self.initial_target_radius*math.cos(self.initial_target_angle)
        
        self.model.body_pos[4][0] = x_pos
        self.model.body_pos[4][1] = y_pos
        self.target_com = self.get_body_com("target")

    def get_initial_polar(self):
        self.initial_target_radius = math.sqrt(pow(self.target_com[0],2)+pow(self.target_com[1],2))
        self.initial_target_angle = math.atan2(self.target_com[1],self.target_com[0])
        self.angular_vel = random.uniform(-1.0,1.0)
    
    #step function for environment
    def step(self, a):
        self.update_goal() # move target

        dist_old = self.calc_dist() # calculate old distance between ee and target
        self.theta_old = np.array(self.sim.data.qpos.flat[:2]) # get old joint positions
        self.e_tank_old = self.e_tank # store current energy level in tank 

        self.do_simulation(a, self.frame_skip) # apply the action to environment
        
        dist_new = self.calc_dist() # calculate new distance between ee and target
        self.theta = np.array(self.sim.data.qpos.flat[:2])# get new joint positions

        self.energy_out(a) # calculate energy going out, energy in tank is updated
        self.energy_in(dist_old,dist_new) # calculate energy coming in , energy in tank is updated

        ob = self._get_obs() # get the sensor value 

        reward_dist = -0.5*dist_old # distance reward 
        reward_energy =self.energy_coming_in - self.energy_going_out # energy reward
        reward = reward_dist + reward_energy - 5*(1 - self.e_tank/self.max_energy) #total reward
        reward_time = 1.0
        
        if self.e_tank <= 0.0:
            reward -= 500 # provide a big negative reward if energy in the tank is 0
            done =True
            print('energy out')
        elif dist_new<0.05:
            reward += (20) # provide positive reward if robot is near the target
            done = False
            print('target reached')
        else:
            done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_energy=reward_energy, reward_time = reward_time, e_out = self.energy_going_out, e_tank=self.e_tank)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    #method to reset the environment for a new episode
    def reset_model(self):
        random_angle = self.np_random.uniform(low=-0.1, high=0.1, size=2)
        qpos =self.init_qpos
        qpos[0] = random_angle[0]
        qpos[1] = random_angle[1]
        while True:
            self.goal = self.np_random.uniform(low=-1.5, high=1.5, size=2)
            if np.linalg.norm(self.goal) < 1.5:
                break
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        #qvel[-2:] = 0
        self.model.body_pos[4][0] = self.goal[0]
        self.model.body_pos[4][1] = self.goal[1]
        self.set_state(qpos, qvel)

        self.e_tank =7.0
        self.energy_going_out=0.0
        self.energy_coming_in=0.0
        
        self.target_com = self.get_body_com("target")
        self.get_initial_polar()
        print("tar",self.target_com)
        self.init_dist = self.calc_dist()
        print("init_dist", self.init_dist)
        self.theta = np.array(self.sim.data.qpos.flat[:2])
        return self._get_obs()
    
    #method to get state for reinforcement learning 
    def _get_obs(self):
        return np.concatenate([
            np.cos(self.theta),
            np.sin(self.theta),
            self.sim.data.qvel.flat[:2],
            self.target_com[:2],
            self.get_body_com("fingertip")[:2],
            [self.e_tank/self.max_energy],
            [self.e_tank_old/self.max_energy],
            self.e_out_vec,
            [self.energy_going_out],
            [self.energy_coming_in]
        ])
