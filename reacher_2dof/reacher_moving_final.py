import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import random


class ReacherMovingFinalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    max_energy = 10.0
    ein_coeff =15

    e_tank = 7
    e_tank_old =  e_tank

    e_out_vec=[0.0,0.0]
    energy_going_out=0.0
    energy_coming_in=0.0

    theta = [0.0,0.0]
    theta_old =[0.0,0.0]
    init_dist=0.0
    target_com=[0.0,0.0,0.0]

    time_step = 0.05
    initial_target_radius= 0.0
    initial_target_angle=0.0
    angular_vel =0.0
    #action = [0.0,0.0]

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_no_friction_no_box.xml', 2)

    def energy_out(self,action):
        d_theta = self.theta-self.theta_old
        self.e_out_vec = action*d_theta
        if self.e_out_vec[0]<=0:
            self.e_out_vec[0] = action[0]*action[0]
        if self.e_out_vec[1]<=0:
            self.e_out_vec[1] = action[1]*action[1]
        self.energy_going_out = sum(self.e_out_vec)
        self.e_tank -= self.energy_going_out

    def energy_in(self,prev_dist,next_dist):
        e_in = self.ein_coeff*(prev_dist-next_dist)
        self.energy_coming_in = e_in
        e_in = max(0.0,e_in)
        e_in = min(e_in,self.max_energy-self.e_tank_old)
        self.e_tank += e_in

    def calc_dist(self):
        vec = self.get_body_com("fingertip")[:2]-self.target_com[:2]
        return  np.linalg.norm(vec)
    
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

    def step(self, a):
        self.update_goal()

        dist_old = self.calc_dist()
        self.theta_old = np.array(self.sim.data.qpos.flat[:2])
        self.e_tank_old = self.e_tank

        self.do_simulation(a, self.frame_skip)
        
        dist_new = self.calc_dist()
        self.theta = np.array(self.sim.data.qpos.flat[:2])

        self.energy_out(a)
        self.energy_in(dist_old,dist_new)

        ob = self._get_obs()

        reward_dist = -0.5*dist_old
        reward_energy =self.energy_coming_in - self.energy_going_out
        reward = reward_dist + reward_energy - 5*(1 - self.e_tank/self.max_energy)
        reward_time = 1.0
        
        if self.e_tank <= 0.0:
            reward -= 500
            done =True
            print('energy out')
        elif dist_new<0.05:
            reward += (20)#+10*self.energy_tank)
            done = False
            print('target reached')
        else:
            done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_energy=reward_energy, reward_time = reward_time, e_out = self.energy_going_out, e_tank=self.e_tank)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

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
        
        '''

        self.box_angles = np.random.uniform(low=-3.14, high=3.14, size = 2)
        self.box_radii = np.random.uniform(low = 0.5, high = 1.5, size = 2)
        self.box_mass = np.random.uniform(low = 1, high = 50, size = 2)

        self.model.body_pos[5][0] = self.box_radii[0]*np.cos(self.box_angles[0])
        self.model.body_pos[5][1] = self.box_radii[0]*np.sin(self.box_angles[0])
        self.model.body_mass[5] = 0.1#self.box_mass[0]

        self.model.body_pos[6][0] = self.box_radii[1]*np.cos(self.box_angles[1])
        self.model.body_pos[6][1] = self.box_radii[1]*np.sin(self.box_angles[1])
        self.model.body_mass[6] = 5#self.box_mass[1]
        qpos[2:4] = [ self.model.body_pos[5][0],  self.model.body_pos[5][1]]
        qpos[8:10] = [ self.model.body_pos[6][0],  self.model.body_pos[6][1]]
        '''
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
