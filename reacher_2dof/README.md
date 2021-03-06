## Experiment 1: A proof of concept
In this experiment a 2 DOF planar robot learns to track a moving target. The energy in the virtual energy tank associated with the 2 motors is always maintained below an emperically set, energy safety threshold of 10J. In the beginning of each trial, the target is initialized with a random angular velocity around the robot and at a random location, hence, the total task energy required for the trial is unknown. However, by virtue of the selective passivity property provided by the framework the robot is able to recharge the energy tank with new energy if it keeps moving towards the target. Now, if fixed obstacles are introduced in the arena and the robot tries to exert force against them, then by virtue of the same selectve passivity property the robot is  unable to release more than 10J of energy in a single timestep thus providing safety.   
![](pics/without_obstacles.gif)
![](pics/with_obstacles.gif)

## Setup
The envronment is adapted from the ['Reacher-v2' environment](https://gym.openai.com/envs/Reacher-v2/) provided by [OpenAI's Gym](https://gym.openai.com/) with some modifications such as offsetting the robot along the Z-axis by 0.2 m and including joint torque losses. 

### Energy Tank Update
Energy tank is initialized at 7J in the beginning of each trial. Maximum permissible energy is set at 10J. Energy flowing out of the tank is calculated using the [energy samplig approach](https://ieeexplore-ieee-org.ezproxy2.utwente.nl/document/8463174), which uses the product of motor torques and corresponding change in joint positions to calculate the energy used by each motor.
The energy flowing in depends on the change in distance between the end effector of the robot and the target.

### RL framework
The state space used for RL algorithm includes information about position, velocity and torques commands corresponding to both joints, current location of the target and energy levels of the tank. The action output consists of the torque commands to both the joints.   
The goal was not to outperform state of the art RL algorithms hence the [DDPG algorithm](https://arxiv.org/abs/1509.02971) was emperically selected with following hyper parameters. More details can be found in section 4 of the [thesis](http://essay.utwente.nl/88729/1/Chaturvedi_MA_EEMCs.pdf).

### How to run
After cloning and installing the dependencies, run the learn.py python file to start learning
```bash
python3 learn.py
```
After training is complete the learned policy can be tested using te infer.py program as:
```bash
python3 infer.py
```
