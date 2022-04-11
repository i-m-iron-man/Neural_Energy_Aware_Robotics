import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Reacher1DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    max_energy = 8
    ein_coeff = 1.5
    
    e_tank = 5
    e_tank_old =  e_tank
    
    energy_going_out = 0.0
    energy_going_out_old = 0.0

    energy_coming_in = 0.0
    energy_coming_in_old = 0.0

    target_theta= [3.14]
    theta = 0.0
    theta_old = 0.0
    
    vel_old = 0.0
    action_old = 0.0

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'reacher1D.xml', 5)
        utils.EzPickle.__init__(self)

    def energy_in(self,prev_dist,next_dist):
        e_in = self.ein_coeff*(prev_dist-next_dist)
        self.energy_coming_in = e_in
        e_in = max(0.0,e_in)
        e_in = min(e_in,self.max_energy-self.e_tank_old)
        self.e_tank += e_in

    def energy_out(self,action):
        d_theta = self.theta-self.theta_old
        e_out = max(3*action[0]*d_theta,0.0)
        if e_out==0.0:
            e_out=9*action[0]*action[0]
        self.energy_going_out = e_out
        self.e_tank -= self.energy_going_out

    def calc_dist(self):
        return abs(np.cos(self.target_theta[0]) - np.cos(self.sim.data.qpos[0]))

    def step(self, action):
        dist_old = self.calc_dist()
        self.theta_old = self.sim.data.qpos[0]
        self.vel_old = self.sim.data.qvel.flat[0]
        self.e_tank_old = self.e_tank
        self.energy_going_out_old = self.energy_going_out
        self.energy_coming_in_old = self.energy_coming_in

        self.do_simulation(action, self.frame_skip)
        
        dist_new = self.calc_dist()
        self.theta = self.sim.data.qpos[0]
        self.energy_out(action)
        self.energy_in(dist_old,dist_new)
        
        ob = self._get_obs()

        self.action_old = action[0]
        
        reward_energy=self.energy_coming_in - self.energy_going_out
        reward_dist=-0.2*dist_old
        tank_reward = 2*(1-(self.e_tank/self.max_energy))
        reward= reward_energy-tank_reward+reward_dist
                    
        if self.e_tank <= 0.0:
            reward -= 500
            done =True
            print('energy out')
        
        elif dist_new<0.01:
            reward += (500)#+10*self.energy_tank)
            done = True
            print('target reached')
        else:
            done = False
        return ob, reward, done, dict(reward_energy = reward_energy, reward_tank = tank_reward, e_out=self.energy_going_out, e_tank=self.e_tank)

    def _get_obs(self):
        return np.concatenate([
            [np.cos(self.theta)],
            [np.sin(self.theta)],
            [np.cos(self.theta_old)],
            [np.sin(self.theta_old)],
            self.sim.data.qvel.flat[:1],
            [self.vel_old],
            np.cos(self.target_theta),
            [self.e_tank/self.max_energy],
            [self.e_tank_old/self.max_energy],
            [self.energy_coming_in],
            [self.energy_coming_in_old],
            [self.energy_going_out],
            [self.energy_going_out_old],
            [self.action_old],
        ])

    def reset_model(self):
        random_angle = [0]#self.np_random.uniform(low=-0.1, high=0.1, size=1)
        qpos =self.init_qpos
        qpos[0] = random_angle[0]
        self.target_theta = [3.14]#self.np_random.uniform(low=2.5, high=3.6, size=1)
        print("target_theta:",self.target_theta)
        target_x = .85*np.cos(self.target_theta[0])
        target_y = .85*np.sin(self.target_theta[0])
        '''
        while True:
            self.box_angle = self.np_random.uniform(low=-3.14, high=3.14, size=1)
            if self.box_angle!=self.target_theta:
                break
        self.model.body_pos[4][0] = 0.7071*np.cos(self.box_angle[0])
        self.model.body_pos[4][1] = 0.7071*np.sin(self.box_angle[0])
        '''
        box1 = self.np_random.uniform(low=-1, high=1, size=1)
        mass = self.np_random.uniform(low=0.2, high=4, size=2)
        '''
        if box1[0]>0:
            #self.model.body_pos[4][0] = 0.5
            #self.model.body_pos[4][1] = 0.5
            self.model.body_pos[4][0] = 0.5
            self.model.body_pos[4][1] = 0.5
        else:
            #self.model.body_pos[4][0] = 0.5
            #self.model.body_pos[4][1] = -0.5
            self.model.body_pos[4][0] = 0.5
            self.model.body_pos[4][1] = 0.5
        qpos[1:3] = [self.model.body_pos[4][0],self.model.body_pos[4][1]]
        '''
        self.model.body_mass[4] = mass[0]
        self.model.body_mass[5] = mass[1]
        print("mass of box1:",self.model.body_mass[4])
        print("mass of box2:",self.model.body_mass[5])
        '''
        while True:
            self.box2 = self.np_random.uniform(low=-1.5, high=1.5, size=2)
            if np.linalg.norm(self.box2) < 1.5 2nd np.any(self.box2 != self.goal) and np.any(self.box2 != self.box1):
                break
        if(abs(self.box1[1])<0.1 and self.box1[0]>=0):
            self.box1[1] +=0.2
        if(abs(self.box2[1])<0.1 and self.box2[0]>=0):
            self.box2[1] +=0.2
        self.model.body_pos[5][0] = self.box1[0]
        self.model.body_pos[5][1] = self.box1[1]
        '''
        self.model.body_pos[3][0] = target_x
        self.model.body_pos[3][1] = target_y
        #qpos[2:4] = self.box2
        qvel = self.init_qvel #+ self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        



        self.e_tank =5.0
        self.e_tank_old = self.e_tank
        self.energy_going_out_old=0.0
        self.energy_going_out=0.0
        self.energy_coming_in_old=0.0
        self.energy_coming_in=0.0
        self.theta = self.sim.data.qpos[0]
        return self._get_obs()