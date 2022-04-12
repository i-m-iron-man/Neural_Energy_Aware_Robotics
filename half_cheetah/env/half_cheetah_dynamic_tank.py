import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

cwd = os.getcwd()

class HalfCheetahDynamictankEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    starting_energy = 40
    ein_coeff = 30
    e_tank = starting_energy
    e_tank_prev =  e_tank
    low_energy_thresh = 0.33
    energy_going_out = 0.0
    e_out_vec = [0.0,0.0,0.0,0.0,0.0,0.0]
    energy_coming_in = 0.0

    low_energy_penalty=20

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, cwd+'/env/half_cheetah.xml', 6)
        utils.EzPickle.__init__(self)

    def energy_in(self,prev_pos,next_pos):
        e_in = self.ein_coeff*(next_pos-prev_pos)
        self.energy_coming_in = e_in
        e_in = max(0.0,e_in)
        self.e_tank += e_in
        self.e_tank = min(self.e_tank,self.starting_energy)

    def energy_out(self,action,prev_theta):
        theta=self.sim.data.qpos[3:]
        d_theta = theta-prev_theta
        self.e_out_vec[0] = max(1.2*action[0]*d_theta[0],0.0)
        self.e_out_vec[1] = max(0.9*action[1]*d_theta[1],0.0)
        self.e_out_vec[2] = max(0.6*action[2]*d_theta[2],0.0)
        self.e_out_vec[3] = max(1.2*action[3]*d_theta[3],0.0)
        self.e_out_vec[4] = max(0.6*action[4]*d_theta[4],0.0)
        self.e_out_vec[5] = max(0.3*action[5]*d_theta[5],0.0)
        if self.e_out_vec[0]==0.0:
            self.e_out_vec[0]=1.44*action[0]*action[0]
        if self.e_out_vec[1]==0.0:
            self.e_out_vec[1]=0.81*action[1]*action[1]
        if self.e_out_vec[2]==0.0:
            self.e_out_vec[2]=0.36*action[2]*action[2]
        if self.e_out_vec[3]==0.0:
            self.e_out_vec[3]=1.44*action[3]*action[3]
        if self.e_out_vec[4]==0.0:
            self.e_out_vec[4]=0.36*action[4]*action[4]
        if self.e_out_vec[5]==0.0:
            self.e_out_vec[5]=0.09*action[5]*action[5]
        self.energy_going_out = sum(self.e_out_vec)
        self.e_tank -= self.energy_going_out

    def tank_change(self,energy_change):
        if self.starting_energy>40 or energy_change>0:
            self.starting_energy += energy_change

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        prev_theta = self.sim.data.qpos[3:]
        self.e_tank_prev = self.e_tank
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        self.energy_out(action,prev_theta)
        self.energy_in(xposbefore,xposafter)
        ob = self._get_obs()
        reward_energy = self.energy_coming_in - self.energy_going_out
        
        reward_survive=0.5
        reward = reward_energy + reward_survive
        reward -= 30*(1-(self.e_tank/self.starting_energy))
        done = False
        if self.e_tank<=0.0:
            done=True
            reward -= 35
        return ob, reward, done, dict(reward_energy=reward_energy, reward_survive=reward_survive, e_out=self.energy_going_out, e_tank=self.e_tank)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.e_out_vec,
            [self.energy_coming_in],
            [self.e_tank_prev/self.starting_energy],
            [self.e_tank/self.starting_energy]
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        #master will update self.etank already
        self.e_tank =self.starting_energy
        self.e_tank_prev = self.starting_energy
        print("energy starting at",self.e_tank)
        self.e_out_vec = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.energy_going_out = 0.0
        self.energy_coming_in = 0.0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5